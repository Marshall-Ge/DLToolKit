from functorch.einops import rearrange
from sympy.matrices.expressions.factorizations import QofQR
from torch import nn
import torch
import math
import torch.nn.functional as F

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

        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

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

        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

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
