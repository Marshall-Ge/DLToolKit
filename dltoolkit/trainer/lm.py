import hydra
from dltoolkit.utils.utils import *
from dltoolkit.datasets.utils import blending_datasets
from dltoolkit.datasets import LMDataset
from dltoolkit.trainer.base_trainer import BaseTrainer

import logging
import math
import torch
from tqdm import tqdm
import os
from transformers.trainer import get_scheduler
import transformers

logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="lm", version_base=None)
def main(config) -> None:

    run_lm(config)

def run_lm(config) -> None:
    # configure strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()
    strategy.print(f"Configs: {config}")

    # configure model
    model = get_local_or_pretrained_model(config, config.model.type)
    strategy.print(model)

    # prepare tokenizer
    tokenizer = get_tokenizer(config)

    # configure optimizer
    optimizer = strategy.create_optimizer(model)

    # prepare for data and dataset
    train_data = blending_datasets(
        config.data.name_or_path,
        config.data.probs,
        strategy,
        config.seed,
        max_count=config.data.max_samples,
        dataset_split=config.data.split,
    )

    train_dataset = LMDataset(
        train_data,
        strategy,
        config.data.max_len,
        tokenizer = tokenizer,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        pin_memory=True,
        shuffle=True,
        collate_fn= transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    if getattr(config.data, "eval_dataset", None):
        eval_data = blending_datasets(
            config.data.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=config.data.eval_split,
        )
    else:
        eval_data = train_data.select(range(int(len(train_data) * 0.01)))

    eval_dataset = LMDataset(
        eval_data,
        strategy,
        config.data.max_len,
        tokenizer=tokenizer,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        pin_memory=True,
        shuffle=False,
        collate_fn = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // config.trainer.train_batch_size
    max_steps = math.ceil(config.trainer.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        config.trainer.lr_scheduler,
        optimizer,
        num_warmup_steps=math.ceil(max_steps * config.trainer.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": config.trainer.learning_rate * 0.1},
    )

    # strategy prepare
    model, optimizer, scheduler, train_dataloader, eval_dataloader = (
        strategy.engine.prepare(model, optimizer, scheduler, train_dataloader, eval_dataloader))

    # load checkpoint
    consumed_samples = 0
    if config.load_checkpoint and os.path.exists(config.ckpt_path):
        _, states = strategy.load_ckpt(model, config.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {config.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(config.save_path, exist_ok=True)

    trainer = LMTrainer(
        model=model,
        strategy=strategy,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        tokenizer=tokenizer,
        max_epochs=config.trainer.max_epochs,
    )

    strategy.print(f"***** Running training *****")
    strategy.print(f"  Num examples = {len(train_dataset)}")
    strategy.print(f"  Num Epochs = {config.trainer.max_epochs}")
    strategy.print(f"  Instantaneous batch size per device = {config.trainer.train_batch_size}")
    strategy.print(f"  Total optimization steps = {max_steps}")

    trainer.fit(config, consumed_samples, num_update_steps_per_epoch)

    # save final model
    model = strategy.engine.unwrap_model(model)
    model.save_pretrained(config.save_path)


class LMTrainer(BaseTrainer):
    """
    Trainer for training a language model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optimizer (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        strategy,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer = None,
        max_epochs: int = 2,
        loss="cross_entropy",
    ) -> None:
        super().__init__(
            model,
            strategy,
            optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            tokenizer,
            max_epochs,
            loss,
        )


    def fit(self, config, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if config.trainer.eval_steps == -1:
            config.trainer.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if config.trainer.save_steps == -1:
            config.trainer.save_steps = float("inf")  # do not save ckpt
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

        self.global_step = consumed_samples // config.trainer.train_batch_size  + 1
        start_epoch = consumed_samples // config.trainer.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * config.trainer.train_batch_size)

        def train(dataloader, step_bar):
            self.model.train()
            for batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                self.strategy.engine.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                # optional info
                logs_dict = {
                    'loss': loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                client_states = {"consumed_samples": self.global_step * config.trainer.train_batch_size}
                self.save_logs_and_checkpoints(config, self.global_step, step_bar, logs_dict, client_states)

                self.global_step += 1

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank0())
        for epoch in range(start_epoch, self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank0(),
            )
            if isinstance(self.strategy, AccelerateStrategy) and epoch == start_epoch and consumed_samples > 0:
                # skip the consumed samples in the first epoch
                skipped_dataloader = self.strategy.engine.skip_first_batches(
                    self.train_dataloader, consumed_samples // config.trainer.train_batch_size
                )
                train(skipped_dataloader, step_bar)
            else:
                train(self.train_dataloader, step_bar)
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank0():
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, config, global_step, step_bar, logs_dict, client_states):
        if global_step % config.trainer.log_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if (
            global_step % config.trainer.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save checkpoint
        if global_step % config.trainer.save_steps == 0 or global_step % self.num_update_steps_per_epoch == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model, tag, config.ckpt_path, config.max_ckpt_num, config.max_ckpt_mem, client_states)

    def evaluate(self, eval_dataloader, global_step):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % global_step,
            disable=not self.strategy.is_rank0(),
        )
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            for batch in eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss_sum += loss.item()
                step_bar.update()

            loss_mean = loss_sum / len(eval_dataloader.dataset)
            perplexity = math.exp(loss_sum)

            bar_dict = {
                "eval_loss": loss_mean,
                "perplexity": perplexity,
            }

            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, global_step)
        self.model.train()  # reset model state


if __name__ == "__main__":
    main()