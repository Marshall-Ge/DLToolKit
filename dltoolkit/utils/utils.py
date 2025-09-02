from dltoolkit.utils.strategy import *
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

def get_strategy(args):

    # TODO: temporary we use accelerate strategy, future we will support FSDP strategy.
    strategy = AccelerateStrategy(args)
    return strategy

def get_local_model(config, model_type='llm'):

    name = config.model.name_or_path
    model = MODEL_REGISTRY.get(name)(config)

    return model