from torch import nn
import torch
import torch.nn.functional as F
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


####################################################################################
# Utils Mechanism: interpolate_pos_embed
####################################################################################

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # cls_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1,orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print(f'reshape pos embedding from {orig_size ** 2:%d} to {new_size ** 2:%d}')
        return new_pos_embed

    else:
        return pos_embed_checkpoint

####################################################################################
# Utils Function:
####################################################################################
@torch.no_grad()
def _load_weights(model, checkpoint_path, prefix=''):
    #TODO: Need Enhancements
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    model.load_state_dict(state_dict, strict=True)

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):

    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')



