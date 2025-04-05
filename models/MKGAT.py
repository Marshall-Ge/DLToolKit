import torch
from torch import nn
import torch.nn.functional as F
from .clip import create_pretrained_clip
from PIL import Image

class GraphAttentionLayer(nn.Module):
    def __init__(self, emb_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., alpha=0.2):
        """
        Note: emb_dim should be 3 * dense_dim
        """
        super().__init__()
        self.num_heads = num_heads
        assert emb_dim % num_heads == 0, 'emb_dim must be divisible by num_heads'
        self.head_dim = emb_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.W = nn.Linear(emb_dim, num_heads * self.head_dim, bias=qkv_bias)
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * self.head_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.skip_W = nn.Linear(emb_dim, 3 * emb_dim)

    def forward(self, eh, er, et, h_id):
        """
        Args:
            eh: input feature [N,C]
            er: input feature [N,C]
            et: input feature [N,C]
            h_id: input feature [N,]
        """
        N = eh.size(0)
        # 合并输入特征
        e = torch.cat([eh, er, et], dim=-1)  # [N, 3 * C]
        # 线性变换
        Wh = self.W(e).view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]

        # 计算注意力系数
        a_input = torch.cat([Wh.repeat(1, 1, N).view(N * N, self.num_heads, self.head_dim),
                             Wh.repeat(N, 1, 1)], dim=-1).view(N, N, self.num_heads, 2 * self.head_dim)
        # [N, N, num_heads, 2 * head_dim]
        e_ij = []
        for head in range(self.num_heads):
            head_a_input = a_input[:, :, head, :]  # [N, N, 2 * head_dim]
            head_e_ij = self.leakyrelu(torch.matmul(head_a_input, self.a).squeeze(-1))  # [N, N]
            e_ij.append(head_e_ij)
        e_ij = torch.stack(e_ij, dim=-1)  # [N, N, num_heads]
        e_ij = e_ij * self.scale  # [N, N, num_heads]
        # 使用 h_id 构建掩码
        mask = (h_id.unsqueeze(0) != h_id.unsqueeze(1)).unsqueeze(-1)
        e_ij = e_ij.masked_fill(mask, float('-inf'))
        print(e_ij)
        # 计算注意力权重
        attn_weights = torch.softmax(e_ij, dim=1)  # [N, N, num_heads]
        attn_weights = self.attn_drop(attn_weights)

        # 计算输出
        Wh_expanded = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, num_heads, head_dim]
        attn_output = attn_weights.unsqueeze(-1) * Wh_expanded  # [N, N, num_heads, head_dim]
        out = attn_output.sum(dim=1)  # [N, num_heads, head_dim]
        out = out.reshape(N, -1)  # [N, num_heads * head_dim]
        out = self.proj(out)
        out = self.proj_drop(out)

        # skip connection
        skip_out = self.skip_W(e)
        skip_out = skip_out + out

        return skip_out

class MMKG_Embedding(nn.Module):
    # h: head node: item_entity
    # r: relation: set(has_image, has_text)
    # t: tail node: multimodal_entity
    def __init__(self, num_items, emb_dim = 256 ,hidden_emb_dim = 64, device = 'cpu', clip_arch = "ViT-B/32"):
        super().__init__()
        self.num_items = num_items
        self.head_embedding = nn.Embedding(num_items, hidden_emb_dim)
        self.relation_embedding = nn.Embedding(2, hidden_emb_dim)

        # tailing_embedding
        # clip settings
        clip_dim_dict = {'ViT-B/32': 512}
        clip_modal, preprocess = create_pretrained_clip(clip_arch, device=device)
        self.image_preprocess = preprocess
        self.clip_modal = clip_modal
        # frozen clip model
        for param in self.clip_modal.parameters():
            param.requires_grad = False

        # dense layer
        self.h_dense = nn.Linear(hidden_emb_dim, emb_dim)
        self.r_dense = nn.Linear(hidden_emb_dim, emb_dim)

        self.image_dense = nn.Linear(clip_dim_dict[clip_arch], emb_dim)
        self.text_dense = nn.Linear(clip_dim_dict[clip_arch], emb_dim)

        self.gat = GraphAttentionLayer(emb_dim * 3)

    def forward(self, input_g):
        item_id = input_g['item_id']
        r_id = input_g['r_id']
        t = input_g['t']

        eh = self.head_embedding(item_id)
        eh = self.h_dense(eh)
        er = self.relation_embedding(r_id)
        er = self.r_dense(er)

        if r_id == 0: # image
            image = t
            image = self.image_preprocess(image).unsqueeze(0)
            image = self.clip_modal.encode_image(image)
            e_t = self.image_dense(image)
        else: # text
            text = t
            text = self.clip_modal.encode_text(text)
            e_t = self.text_dense(text)

        attn_out = self.gat(eh, er, e_t, item_id)
        return attn_out, er, e_t