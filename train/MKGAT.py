import torch

from builds import build_amazon_dataloader
from models import MMKG_Embedding
import argparse
from utils.config import parse_args


def kg_loss():
    pass


if __name__ == '__main__':
    cfg = parse_args("MKGAT")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = build_amazon_dataloader(cfg.dataset_path, batch_size= cfg.batch_size, device=device)
    for batch in data_loader:
        print(batch['item_id'].shape)
        print(batch['r_id'].shape)
        print(batch['t'].shape)