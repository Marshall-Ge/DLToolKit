import hydra
from dltoolkit.utils.utils import *
from dltoolkit.datasets.utils import blending_datasets
from dltoolkit.datasets import CLDataManager
from dltoolkit.trainer.base_trainer import BaseTrainer

import logging
import math
import torch
from tqdm import tqdm
import os
from transformers.trainer import get_scheduler
logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="cl", version_base=None)
def main(config) -> None:

    run_cl(config)

def run_cl(config):
    # config strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()
    strategy.print(f"Configs: {config}")

    # configure model
    model = get_local_or_pretrained_model(config, 'cl')
    strategy.print(model)

    # prepare transform if needed
    transform = get_image_transform(config)

    # configure optimizer
    optimizer = strategy.create_optimizer(model)







class CLTrainer(BaseTrainer):
    pass

if __name__ == "__main__":
    main()