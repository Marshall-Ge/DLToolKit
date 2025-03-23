import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.nn import MultiheadAttention
from models import *
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

input_img = torch.randn((3, 3, 512, 512)).to(device)
pad_mask = pad_mask.unsqueeze(1).expand(batch_size, 512*512, seq_len)

cross_multi_att = CrossMultiAttention(emb_dim, in_channel=3, num_heads=8, att_dropout=0.1).to(device)
outputs, att_weights = cross_multi_att(input_img, inputs, pad_mask)
print(outputs.shape)
print(att_weights.shape)