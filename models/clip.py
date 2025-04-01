from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip



def create_pretrained_clip(
    model_name: str = "ViT-B-32",
    device: str = "cpu",
    jit: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """
    Create a pretrained CLIP model.

    Args:
        model_name (str): Name of the CLIP model to create.
        device (str): Device to load the model on.
        jit (bool): Whether to use JIT compilation.

    Returns:
        Tuple[nn.Module, nn.Module]: The text and image encoders.
    """
    clip_model, preprocess = clip.load(model_name, device=device, jit=jit)
    return clip_model, preprocess