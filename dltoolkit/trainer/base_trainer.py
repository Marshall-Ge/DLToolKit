from abc import ABC, abstractmethod
import torch
import os

from dltoolkit.utils.strategy import BaseStrategy


class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        strategy: BaseStrategy,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer = None,
        max_epochs: int = 2,
        loss="cross_entropy",
    ) -> None:
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.config = strategy.config

        if loss == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError()

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.config.tracker.use_wandb and self.strategy.is_rank0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.config.tracker.use_wandb)
            wandb.init(
                entity=strategy.config.tracker.wandb_org,
                project=strategy.config.tracker.wandb_project,
                group=strategy.config.tracker.wandb_group,
                name=strategy.config.tracker.wandb_run_name,
                config=strategy.config.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.config.tracker.use_tensorboard and self._wandb is None and self.strategy.is_rank0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.config.tracker.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.config.tracker.use_tensorboard, strategy.config.tracker.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

        self.num_update_steps_per_epoch = None

    @abstractmethod
    def fit(self, config, consumed_samples, num_update_steps_per_epoch):
        pass