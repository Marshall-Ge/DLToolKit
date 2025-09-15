from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dltoolkit.utils.strategy import *
from fvcore.common.registry import Registry
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
    model = None

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

    elif model_type == 'img_cls':
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