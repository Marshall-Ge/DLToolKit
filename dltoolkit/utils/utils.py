from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dltoolkit.utils.strategy import *
from fvcore.common.registry import Registry

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

def get_local_or_pretrained_model(config,
                                  model_type='lm',
                                  bf16=True,
                                  load_in_4bit=False,
                                  lora_rank=0,
                                  lora_alpha=16,
                                  target_modules=None,
                                  lora_dropout=0,
                                  attn_implementation="flash_attention_2",
                                  device_map=None,
                                  packing_samples=False,
                                  **kwargs,
                                  ):
    model = None

    if model_type == 'causal_lm':
        config = AutoConfig.from_pretrained(config.model.name_or_path, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        # TODO: support lora & customize model
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            packing_samples=packing_samples,
            **kwargs,
        )

    elif model_type == 'img_cls':
        name = config.model.name_or_path
        model = MODEL_REGISTRY.get(name)(config)

    return model

def get_tokenizer(config, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_image_transform(config):
    if config.data.img_transform.type == 'default':
        return None
    else:
        raise NotImplementedError