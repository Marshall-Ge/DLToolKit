import hydra
from dltoolkit.utils.utils import *
from dltoolkit.datasets.utils import blending_datasets
from dltoolkit.datasets import ImgTxtPairDataset
from dltoolkit.trainer.base_trainer import BaseTrainer

import logging
import math
import torch
from tqdm import tqdm
import os
from transformers.trainer import get_scheduler
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="img_cls", version_base=None)
def main(config) -> None:

    run_img_cls(config)


def run_img_cls(config) -> None:
    # configure strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()

    # configure model
    # TODO: Support customize model now, plan to support hf models
    model = get_local_model(config, 'img_cls')
    strategy.print(model)

    # prepare tokenizer if needed
    tokenizer = None
    # prepare transform if needed
    transform = get_image_transform(config)

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

    train_dataset = ImgTxtPairDataset(
        train_data,
        strategy,
        config.data.max_len,
        input_template=config.data.input_template,
        tokenizer = tokenizer,
        transform=transform,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        True,
        True,
        collate_fn=train_dataset.collate_fn,
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

    eval_dataset = ImgTxtPairDataset(
        eval_data,
        strategy,
        config.data.max_len,
        input_template=config.data.input_template,
        tokenizer=tokenizer,
        transform=transform,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        True,
        False,
        collate_fn=eval_dataset.collate_fn,
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

    trainer = ImgClsTrainer(
        model=model,
        strategy=strategy,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        tokenizer=tokenizer,
        max_epochs=config.trainer.max_epochs,
        loss=config.trainer.loss,
    )

    trainer.fit(config, consumed_samples, num_update_steps_per_epoch)



class ImgClsTrainer(BaseTrainer):
    """
    Trainer for training a image classification model.

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

        # TODO: Restore step and start_epoch
        # global_step = consumed_samples // config.trainer.train_batch_size * self.strategy.accumulated_gradient + 1
        # start_epoch = consumed_samples // config.trainer.train_batch_size // num_update_steps_per_epoch
        # consumed_samples = consumed_samples % (num_update_steps_per_epoch * config.trainer.train_batch_size)

        start_epoch = 0
        global_step = 0

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank0())
        for epoch in range(start_epoch, self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank0(),
            )
            self.model.train()
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                img, label = batch['image'], batch['text_or_label']
                outputs = self.model(img)
                loss = self.loss_fn(outputs, label)
                self.strategy.engine.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                # optional info
                logs_dict = {
                    'loss' : loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                if global_step % config.trainer.log_steps == 0:
                    client_states = {"consumed_samples": global_step * config.trainer.train_batch_size}
                    self.save_logs_and_checkpoints(config, global_step, step_bar, logs_dict, client_states)

                global_step += 1

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
        # if global_step % config.trainer.save_steps == 0:
        #     tag = f"global_step{global_step}"
        #     self.strategy.save_ckpt(
        #         self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
        #     )
        #     if self.save_hf_ckpt:
        #         save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
        #         self.strategy.save_model(self.model, self.tokenizer, save_path)


    def evaluate(self, eval_dataloader, global_step):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % global_step,
            disable=not self.strategy.is_rank0(),
        )
        self.model.eval()
        total = 0
        with torch.no_grad():
            acc_num = 0
            loss_sum = 0
            for batch in eval_dataloader:
                img, label = batch['image'], batch['text_or_label']
                outputs = self.model(img)
                loss = self.loss_fn(outputs, label)
                preds = torch.argmax(outputs, dim=-1)
                preds, label = self.strategy.engine.gather_for_metrics((preds, label))
                total += len(label)
                acc_num += (preds== label).sum().item()
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc_num / len(eval_dataloader.dataset)
            loss_mean = loss_sum / len(eval_dataloader.dataset)
            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
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