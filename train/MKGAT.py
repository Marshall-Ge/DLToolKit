import torch
import torch.nn as nn
import torch.nn.functional as F

from buildDatasets import build_amazon_dataloader
from models import MMKG_Embedding
import argparse
from utils.config import parse_args


class KnowledgeGraphEmbeddingLoss(nn.Module):
    def __init__(self):
        super(KnowledgeGraphEmbeddingLoss, self).__init__()

    def forward(self, eh, er, et, et_prime):
        # Compute the score for valid triplet (h, r, t)
        score_positive = torch.norm(eh + er - et, p=2, dim=1)
        # Compute the score for broken triplet (h, r, t')
        score_negative = torch.norm(eh + er - et_prime, p=2, dim=1)
        # Compute the pairwise ranking loss
        loss = -torch.mean(F.logsigmoid(score_negative - score_positive))
        return loss

def sample_negative_triplet(batch_data):
    #TODO: Implement the negative sampling logic
    return batch_data, batch_data

def train_one_epoch():
    for batch_data in data_loader:

        positive_triplet, negative_triplet = sample_negative_triplet(batch_data)
        eh, er, et = model(positive_triplet)
        _, _, et_prime = model(negative_triplet)



if __name__ == '__main__':
    cfg = parse_args("MKGAT")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_loader = build_amazon_dataloader(
        cfg.dataset_path,
        batch_size= cfg.batch_size,
        device=device,
        need_preprocess= cfg.need_preprocess,
        sample_number= cfg.sample_num,
    )

    with open(f'{cfg.dataset_path}/parent_asin_mapping.txt', 'r') as file:
        num_items = sum(1 for line in file)

    model = MMKG_Embedding(num_items).to(device)
    kg_loss = KnowledgeGraphEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for epoch in range(cfg.epochs):
        train_one_epoch()
