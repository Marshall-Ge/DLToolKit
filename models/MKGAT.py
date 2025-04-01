import torch.nn as nn
from entity_encoder import EntityEncoder
from attention_layer import AttentionLayer

class MKGAT(nn.Module):
    def __init__(self, user_item_graph, knowledge_graph):
        super(MKGAT, self).__init__()
        self.user_item_graph = user_item_graph
        self.knowledge_graph = knowledge_graph
        self.entity_encoder = EntityEncoder()
        self.attention_layer = AttentionLayer()

    def forward(self):
        entity_embeddings = self.entity_encoder(self.knowledge_graph)
        updated_embeddings = self.attention_layer(self.user_item_graph, entity_embeddings)
        return updated_embeddings