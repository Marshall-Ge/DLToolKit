import torch
from torch import nn
import torch.nn.functional as F
from .clip import create_pretrained_clip
from PIL import Image

class MMKG_Embedding(nn.Module):
    # h: head node: user_entity, item_entity
    # r: relation: set(rating, has_image, has_text)
    # t: tail node: multimodal_entity
    def __init__(self, num_users, num_items, emb_dim, device = 'cpu'):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.head_embedding = nn.Embedding(num_users + num_items, emb_dim)
        self.relation_embedding = nn.Embedding(3, emb_dim)

        # tailing_embedding
        self.rating_embedding = nn.Embedding(5, emb_dim)

        clip_modal, preprocess = create_pretrained_clip('ViT-B-32', device=device)
        self.image_preprocess = preprocess
        self.image_encoder = clip_modal.encode_image
        self.text_encoder = clip_modal.encode_text

        # dense layer
        # TODO: to finish

    def forward(self, graph):
        # graph: (h, r, t)
        h, r, t = graph
        # h: [batch_size, num_users + num_items]
        # r: [batch_size, 3]
        # t: [batch_size, multimodal_entity]

        # head embedding
        head_emb = self.head_embedding(h)
        relation_emb = self.relation_embedding(r)
        tail_emb = self.rating_embedding(t)

        # multimodal embedding
        multimodal_emb = torch.cat((head_emb, relation_emb), dim=1) + tail_emb

        return multimodal_emb