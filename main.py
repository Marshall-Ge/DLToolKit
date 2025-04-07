import yaml
import subprocess
from utils import parse_args


def print_usage():
    pass

cfg = parse_args("Main")

if cfg.mode == 'train':
    if cfg.model_name == 'mkgat':
        if cfg.dataset_name not in ['amazon-review-2023']:
            raise ValueError(f"Unsupported dataset for {cfg.model_name}: {cfg.dataset_name}")
        subprocess.run(['python', 'train/MKGAT.py'])
    else:
        print_usage()
        raise ValueError(f"Unsupported model: {cfg.model_name}")