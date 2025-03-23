from functorch.einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F


####################################################################################
# Utils Network: MLP
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

####################################################################################
# Attention Components: Attention, CrossAttention, CrossMultiAttention
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



