from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dltoolkit.utils.strategy import *
from fvcore.common.registry import Registry
import torch
import numpy as np
from scipy.stats import gaussian_kde
from accelerate import Accelerator, load_checkpoint_and_dispatch
from typing import Union, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

def get_strategy(args):

    # TODO: temporary we use accelerate default strategy, future we will support FSDP strategy.
    strategy = AccelerateStrategy(args)
    return strategy

def get_local_or_pretrained_model(cfg,
                                  model_type='lm',
                                  bf16=True,
                                  load_in_4bit=False,
                                  lora_rank=0,
                                  lora_alpha=16,
                                  target_modules=None,
                                  lora_dropout=0,
                                  attn_implementation="flash_attention_2",
                                  device_map=None,
                                  **kwargs,
                                  ):

    if model_type == 'causal_lm':
        config = AutoConfig.from_pretrained(cfg.model.name_or_path, trust_remote_code=True)

        # ensure the platform has cuda
        if torch.cuda.is_available():
            try:
                import flash_attn
                if attn_implementation:
                    assert attn_implementation in ["flash_attention_2", "flash_attention", "triton", "cutlass"], \
                        f"attn_implementation should be one of ['flash_attention_2', 'flash_attention', 'triton', 'cutlass'], but got {attn_implementation}"
                else:
                    attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = None
                logger.warning('flash_attn is not installed, will use default attention implementation')
        else:
            attn_implementation = None
        config._attn_implementation = attn_implementation
        # TODO: support lora & customize model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path,
            config=config,
            trust_remote_code=True,
            dtype=torch.bfloat16 if bf16 else torch.float32,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            **kwargs,
        )

    else:
        name = cfg.model.name_or_path
        model = MODEL_REGISTRY.get(name)(cfg)

    return model

def get_tokenizer(cfg, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_image_transform(cfg):
    if cfg.data.img_transform.type == 'default':
        return None
    else:
        raise NotImplementedError

def compute_model_kl_divergence(
        model_dir1: Union[str, os.PathLike],
        model_dir2: Union[str, os.PathLike],
        model_1: torch.nn.Module = None,
        model_2: torch.nn.Module = None,
        use_kde: bool = False,
        skipped_layers: Optional[List[str]] = None
):
    """
    Compute KL divergence between parameters of two models saved with Accelerator.save_model (supports sharded models)

    Args:
        model_dir1: Directory containing the first saved model (save_directory from Accelerator.save_model)
        model_dir2: Directory containing the second saved model
        model: Model with structure matching the saved models
        use_kde: Whether to use Kernel Density Estimation for distribution approximation (False uses normal distribution)
        skipped_layers: List of layer name prefixes to skip (e.g., ["fc", "layer4."] skips fc layers and layer4)

    Returns:
        kl_results: Dictionary mapping parameter names to their KL divergence values (skipped layers excluded)
        mean_kl: Average KL divergence across all non-skipped parameters
    """
    skipped_layers = skipped_layers or []

    # Initialize Accelerator (used for consistent model loading)
    accelerator = Accelerator()

    # Helper function to load model and extract state dictionary
    def load_model_state_dict(save_dir, model):
        # Load checkpoint (automatically handles sharded files)
        model = load_checkpoint_and_dispatch(
            model,
            save_dir,
            no_split_module_classes=getattr(model, "no_split_module_classes", None)  # For Transformer-style models
        )
        # Ensure model is on CPU with consolidated parameters
        model = model.to("cpu")
        return model.state_dict()

    try:
        # Load state dictionaries for both models
        state_dict1 = load_model_state_dict(model_dir1, model_1)
        state_dict2 = load_model_state_dict(model_dir2, model_2)
    except Exception as e:
        raise ValueError(f"Failed to load models: {e}")

    # Verify parameter names match between models
    params1 = set(state_dict1.keys())
    params2 = set(state_dict2.keys())
    if params1 != params2:
        raise ValueError(f"Model parameters mismatch: differing keys {params1.symmetric_difference(params2)}")

    kl_results = {}

    for param_name in state_dict1.keys():
        if any(param_name.startswith(prefix) for prefix in skipped_layers):
            continue

        # Extract parameters and flatten to 1D arrays
        param1 = state_dict1[param_name].cpu().detach().numpy().flatten()
        param2 = state_dict2[param_name].cpu().detach().numpy().flatten()

        # Handle edge case of empty parameters (theoretically shouldn't occur)
        if len(param1) == 0 or len(param2) == 0:
            kl_results[param_name] = 0.0
            continue

        # Calculate KL divergence (P: model1 distribution, Q: model2 distribution)
        if use_kde:
            # Kernel Density Estimation for more accurate distribution approximation
            try:
                # Add small noise to prevent numerical singularities in KDE
                param1_noise = param1 + 1e-10 * np.random.randn(len(param1))
                param2_noise = param2 + 1e-10 * np.random.randn(len(param2))

                kde_p = gaussian_kde(param1_noise)
                kde_q = gaussian_kde(param2_noise)

                # Create common evaluation grid
                grid_min = min(param1.min(), param2.min())
                grid_max = max(param1.max(), param2.max())
                grid = np.linspace(grid_min, grid_max, 1000)  # Adjust samples for precision/speed

                p = kde_p(grid)
                q = kde_q(grid)

                # Prevent log(0) and negative values
                p = np.clip(p, 1e-10, None)
                q = np.clip(q, 1e-10, None)

                # Numerical integration to compute KL divergence
                kl = np.trapz(p * np.log(p / q), grid)
            except:
                # Fallback to normal distribution approximation if KDE fails
                kl = normal_kl(param1, param2)
        else:
            # Use normal distribution approximation (assumes parameters ~ N(μ, σ²))
            kl = normal_kl(param1, param2)

        kl_results[param_name] = kl

    # 计算平均KL散度（仅包含未跳过的层）
    mean_kl = np.mean(list(kl_results.values())) if kl_results else 0.0
    return kl_results, mean_kl


def normal_kl(param_p: np.ndarray, param_q: np.ndarray) -> float:
    """
    Calculate KL divergence between two normal distributions: KL(N(μ1,σ1²) || N(μ2,σ2²))

    Args:
        param_p: Samples from distribution P
        param_q: Samples from distribution Q

    Returns:
        kl_value: KL divergence value
    """
    mu1, sigma1 = np.mean(param_p), np.var(param_p) + 1e-10  # Add epsilon to prevent division by zero
    mu2, sigma2 = np.mean(param_q), np.var(param_q) + 1e-10
    return np.log(np.sqrt(sigma2 / sigma1)) + (sigma1 + (mu1 - mu2) **2) / (2 * sigma2) - 0.5

if __name__ == "__main__":
    import transformers
    transformers.Trainer