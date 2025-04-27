from functools import partial
from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
from models.utils import _load_weights


####################################################################################
# Utils Network: MLP, TransformerBlock
####################################################################################
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


####################################################################################
# Attention Components: Attention, CrossAttention, CrossMultiAttention，AttentionPool2d
####################################################################################
class Attention(nn.Module):
    """ A basic multi-head attention layer
    """
    def __init__(self, emb_dim, num_heads=8,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert emb_dim % num_heads == 0, 'emb_dim must be divisible by num_heads'
        head_dim = emb_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossMultiAttention(nn.Module):
    def __init__(self, emb_dim, in_channel, num_heads, att_dropout=0.0):
        super(CrossMultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.in_channel = in_channel
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, 'emb_dim must be divisible by num_heads'
        self.head_dim = emb_dim // num_heads

        self.Wk = nn.Linear(emb_dim, emb_dim,bias=False)
        self.Wq = nn.Linear(emb_dim, emb_dim,bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim,bias=False)

        self.proj_in = nn.Conv2d(in_channel, emb_dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(emb_dim, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, content,pad_mask=None):
        """
        x : (batch_size, c, h, w)
        content : (batch_size, seq_len, emb_dim)
        pad_mask : (batch_size, h * w, seq_len)
        """
        b, c, h, w = x.shape
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        Q = self.Wq(x)
        K = self.Wk(content)
        V = self.Wv(content)

        # Split into multiple heads
        Q = Q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # (b, num_heads, h*w, head_dim)
        K = K.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # (b, num_heads, seq_len, head_dim)
        V = V.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # (b, num_heads, seq_len, head_dim)

        # Attention
        att_weights = torch.einsum("bnid, bnjd -> bnij", Q, K)
        att_weights = att_weights * self.scale
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1) # (b, num_heads, h*w, seq_len)
        if self.att_dropout > 0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)

        out = torch.einsum("bnij, bnjd -> bnid", att_weights, V) # (b, num_heads, h * w, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.emb_dim) # (b, h * w, emb_dim)

        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        out = self.proj_out(out)

        return out, att_weights

class CrossAttention(nn.Module):
    def __init__(self, emb_dim, in_channel, att_dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.att_dropout = att_dropout
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channel, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        self.proj_out = nn.Conv2d(emb_dim, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, content, pad_mask=None):
        """
        :param x: (batch_size, c, h, w)
        :param content: (batch_size, seq_len, emb_dim)
        :param pad_mask: (batch_size, h * w, seq_len)
        :return:
        """
        b, c, h, w = x.shape
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        Q = self.Wq(x)
        K = self.Wk(content)
        V = self.Wv(content)

        # Attention
        att_weights = torch.einsum("bid, bjd -> bij", Q, K) # (b, h*w ,seq_len)
        att_weights = att_weights * self.scale
        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        if self.att_dropout > 0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)

        out = torch.einsum("bij, bjd -> bid", att_weights, V) # (b, h*w, emb_dim)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        out = self.proj_out(out)
        return out, att_weights

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.attention = Attention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        # TODO: Need to check the transpose
        x = self.attention(x.transpose(0, 1))  # Transpose to (N, HW+1, C) for Attention

        x = self.c_proj(x[:, 0])  # Only use the pooled output (first token)
        return x



####################################################################################
# Transformer Components: Transformer, Vit
####################################################################################


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage, default settings for ImageNet.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, ckpt_layer=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,patch_size=patch_size,in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i >= depth - ckpt_layer)
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, patch_size ** 2, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, patch_size ** 2 + 1, embed_dim)
        x = x + self.pos_embed[:, :(x.shape[1]), :]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk==i)
        x = self.norm(x)
        return x

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix=prefix)

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Mock data
    batch_size = 3
    seq_len = 5
    emb_dim = 64
    vocab_size = 301
    input_ids = torch.tensor([[100, 200, 300, 300, 0],
                              [22, 33, 44, 0, 0],
                              [66, 55, 66, 30, 0]], dtype=torch.long).to(device)
    pad_mask = input_ids.eq(0)
    emb_layer = nn.Embedding(vocab_size, emb_dim).to(device)
    inputs = emb_layer(input_ids)

    input_img = torch.randn((3, 3, 224, 224)).to(device)
    pad_mask = pad_mask.unsqueeze(1).expand(batch_size, 224 * 224, seq_len)

    def test_cross_attention():
        cross_att = CrossAttention(emb_dim, in_channel=3, att_dropout=0.1).to(device)
        outputs, att_weights = cross_att(input_img, inputs, pad_mask)
        assert outputs.shape == (batch_size, 3, 224, 224)
        assert att_weights.shape == (batch_size, 224 * 224, seq_len)
    test_cross_attention()

    def test_cross_multi_attention():
        cross_multi_att = CrossMultiAttention(emb_dim, in_channel=3, num_heads=8, att_dropout=0.1).to(device)
        outputs, att_weights = cross_multi_att(input_img, inputs, pad_mask)
        assert outputs.shape == (batch_size, 3, 224, 224)
        assert att_weights.shape == (batch_size, 8, 224 * 224, seq_len)
    test_cross_multi_attention()

    def test_vit():
        vit = VisionTransformer(drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1).to(device)
        outputs = vit(input_img)
        assert outputs.shape == (batch_size, 197 ,768)
    test_vit()
