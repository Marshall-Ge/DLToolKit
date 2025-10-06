import hydra
from dltoolkit.utils.utils import *
from dltoolkit.datasets.utils import blending_datasets
from dltoolkit.datasets import ImgTxtPairDataset, CLDataManager, DummyDataset
from dltoolkit.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import logging
import math
import torch
from tqdm import tqdm
import os
from transformers.trainer import get_scheduler
logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="cl_img_cls", version_base=None)
def main(config) -> None:

    run_cl_img_cls(config)


def run_cl_img_cls(config) -> None:
    # configure strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()
    strategy.print(f"Configs: {config}")

    # configure model
    model = get_local_or_pretrained_model(config, 'img_cls')
    strategy.print(model)

    # prepare tokenizer if needed
    tokenizer = None
    # prepare transform if needed
    transform = get_image_transform(config)

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

    # prepare data manager
    train_manager = CLDataManager(train_dataset, config.trainer.init_cls, config.trainer.increment, strategy)

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
    eval_manager = CLDataManager(eval_dataset, config.trainer.init_cls, config.trainer.increment, strategy)

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // train_manager.nb_tasks  // config.trainer.train_batch_size
    max_steps = math.ceil(config.trainer.max_epochs * num_update_steps_per_epoch * train_manager.nb_tasks)

    # strategy prepare
    model.construct_net(train_manager.nb_classes)
    model = (strategy.engine.prepare(model))

    # load checkpoint
    consumed_samples = 0
    if config.load_checkpoint and os.path.exists(config.ckpt_path):
        _, states = strategy.load_ckpt(model, config.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {config.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(config.save_path, exist_ok=True)

    trainer = CLImgClsTrainer(
        model=model,
        strategy=strategy,
        tokenizer=tokenizer,
        max_epochs=config.trainer.max_epochs,
        loss=config.trainer.loss,
        train_manager=train_manager,
        eval_manager=eval_manager,
    )

    strategy.print(f"***** Running training *****")
    strategy.print(f"  Num examples = {len(train_dataset)}")
    strategy.print(f"  Num Epochs = {config.trainer.max_epochs}")
    strategy.print(f"  Instantaneous batch size per device = {config.trainer.train_batch_size}")
    strategy.print(f"  Total optimization steps = {max_steps}")
    strategy.print(f"  Total tasks = {train_manager.nb_tasks}")
    strategy.print(f"  Training classes per task = {train_manager.nb_classes}")

    trainer.fit(config, consumed_samples, num_update_steps_per_epoch)

    # save final model
    strategy.engine.save_model(model, config.save_path)



class CLImgClsTrainer(BaseTrainer):
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
        optimizer = None,
        train_dataloader = None,
        eval_dataloader = None,
        scheduler = None,
        tokenizer = None,
        max_epochs: int = 2,
        loss="cross_entropy",
        capacity = 10000,
        train_manager = None,
        eval_manager = None,
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
        self.capacity = capacity
        self.train_manager = train_manager
        self.eval_manager = eval_manager
        self.reply_loader = None
        self.buffer = []

    def fit(self, config, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if config.trainer.eval_steps == -1:
            config.trainer.eval_steps = num_update_steps_per_epoch * config.trainer.max_epochs  # Evaluate once per task
        if config.trainer.save_steps == -1:
            config.trainer.save_steps = float("inf")  # do not save ckpt
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.num_update_steps_per_task = num_update_steps_per_epoch * config.trainer.max_epochs

        self.global_step = consumed_samples // config.trainer.train_batch_size  + 1
        start_task_id = consumed_samples // config.trainer.train_batch_size // self.num_update_steps_per_task
        start_epoch = consumed_samples // config.trainer.train_batch_size % self.num_update_steps_per_task // self.num_update_steps_per_epoch
        consumed_samples = consumed_samples % (self.num_update_steps_per_task * config.trainer.train_batch_size)

        def train(dataloader,task_id,step_bar=None):
            for batch in dataloader:
                self.optimizer.zero_grad()
                img, label = batch['data'], batch['label']
                if self.replay_loader:
                    try:
                        replay_batch = next(replay_iter)
                        replay_img, replay_labels = replay_batch['data'], replay_batch['label']
                    except:
                        replay_iter = iter(replay_loader)
                        replay_batch = next(replay_iter)
                        replay_img, replay_labels = replay_batch['data'], replay_batch['label']

                    img = torch.cat([img, replay_img.to(img.device)])
                    label = torch.cat([label, replay_labels.to(label.device)])


                outputs = self.model(img, task_id)

                loss = self.loss_fn(outputs, label)
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
                self.save_logs_and_checkpoints(config, self.global_step, task_id, logs_dict, client_states)

                self.global_step += 1

        task_bar = tqdm(range(start_task_id, self.train_manager.nb_tasks), desc="Train task", disable=not self.strategy.is_rank0())
        for task_id in range(start_task_id, self.train_manager.nb_tasks):
            self._freeze_before_increment_training(task_id)
            self.refresh_optimizer(task_id)
            start = sum(self.train_manager._increments[:task_id])
            end = start + self.train_manager._increments[task_id]
            current_classes = list(range(start, end))
            train_ds = self.train_manager.get_dataset(current_classes)
            train_dataloader = self.strategy.setup_dataloader(
                train_ds,
                pin_memory=True,
                shuffle=True,
            )
            eval_ds = self.eval_manager.get_dataset(current_classes)
            eval_dataloader = self.strategy.setup_dataloader(
                eval_ds,
                pin_memory=True,
                shuffle=False,
            )
            all_learned_classes = list(range(end))
            all_eval_ds = self.eval_manager.get_dataset(all_learned_classes)
            all_eval_dataloader = self.strategy.setup_dataloader(
                all_eval_ds,
                pin_memory=True,
                shuffle=False,
            )

            replay_loader = self.get_loader(int(self.config.trainer.train_batch_size) // 2)

            self.train_dataloader, self.replay_loader, self.eval_dataloader, self.all_eval_dataloader = self.strategy.engine.prepare(
                train_dataloader, replay_loader, eval_dataloader, all_eval_dataloader)

            # train epochs
            epoch_bar = tqdm(range(start_epoch, self.epochs), desc=f"Train epoch of task {task_id}", disable=not self.strategy.is_rank0())
            for epoch in range(start_epoch, self.epochs):
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
                    train(skipped_dataloader,task_id,step_bar)
                else:
                    train(self.train_dataloader, task_id, step_bar)
                epoch_bar.update()
            self.add(train_ds)
            task_bar.update()

        if self._wandb is not None and self.strategy.is_rank0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank0():
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, config, global_step, task_id, logs_dict, client_states):
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
            global_step % config.trainer.eval_steps == 0
        ) and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step, log_tag = 'current_task', task_id = task_id)
                self.evaluate(self.all_eval_dataloader, global_step, log_tag = 'all_learned_task', task_id=task_id)


        # save checkpoint
        if global_step % config.trainer.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model, tag, config.ckpt_path, config.max_ckpt_num, config.max_ckpt_mem, client_states)

    def evaluate(self, eval_dataloader, global_step, log_tag, task_id):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % global_step,
            disable=not self.strategy.is_rank0(),
        )
        self.model.eval()
        total = 0
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            acc_num = 0
            loss_sum = 0
            for batch in eval_dataloader:
                img, label = batch['data'], batch['label']
                outputs = self.model(img, task_id)
                loss = self.loss_fn(outputs, label)
                preds = torch.argmax(outputs, dim=-1)
                preds, label = self.strategy.engine.gather_for_metrics((preds, label))
                total += len(label)
                acc_num += (preds== label).sum().item()
                loss_sum += loss.item()
                true_labels.extend(label.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                step_bar.update()

            acc_mean = acc_num / len(eval_dataloader.dataset)
            loss_mean = loss_sum / len(eval_dataloader.dataset)
            bar_dict = {
                f"eval_loss/{log_tag}": loss_mean,
                f"acc_mean/{log_tag}": acc_mean,
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

                    if self.config.tracker.visualize_confusion_matrix and log_tag == "all_learned_task":
                        import sklearn.metrics
                        import matplotlib.pyplot as plt
                        confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, pred_labels, normalize='true')

                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot(ax=ax)
                        plt.title('Confusion Matrix')
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')

                        # Log confusion matrix image to TensorBoard
                        self._tensorboard.add_figure(f'eval/confusion_matrix/{log_tag}', fig, global_step=global_step)


        self.model.train()  # reset model state

    def add(self, dataset):
        import random
        samples = [dataset[i] for i in random.sample(range(len(dataset)), min(1000, len(dataset)))]
        self.buffer.extend(samples)
        if len(self.buffer) > self.capacity:
            random.shuffle(self.buffer)
            self.buffer = self.buffer[:self.capacity]

    def get_loader(self, batch_size):
        if not self.buffer or self.config.trainer.capacity == 0:
            return None
        datas = [item['data'] for item in self.buffer]
        labels = [item['label'] for item in self.buffer]
        dataset = DummyDataset(datas, labels)
        return self.strategy.setup_dataloader(dataset,batch_size,True,True)

    def refresh_optimizer(self, task_id):
        # re-define optimizer for new added parameters
        if task_id == 0:
            self.optimizer = self.strategy.create_optimizer(self.model)
            self.scheduler = get_scheduler(
                self.config.trainer.lr_scheduler,
                self.optimizer,
                num_warmup_steps=math.ceil(self.config.trainer.max_epochs * self.num_update_steps_per_epoch * self.config.trainer.lr_warmup_ratio),
                num_training_steps=math.ceil(self.config.trainer.max_epochs * self.num_update_steps_per_epoch),
                scheduler_specific_kwargs={"min_lr": self.config.trainer.learning_rate * 0.1},
            )
        else:
            adapter = getattr(self.model, f'adapter_{task_id}_fc', None)
            self.optimizer = self.strategy.create_optimizer(adapter, lr=self.config.trainer.inc_learning_rate)
            self.scheduler = get_scheduler(
                self.config.trainer.lr_scheduler,
                self.optimizer,
                num_warmup_steps=math.ceil(self.config.trainer.max_epochs * self.num_update_steps_per_epoch * self.config.trainer.lr_warmup_ratio),
                num_training_steps=math.ceil(self.config.trainer.max_epochs * self.num_update_steps_per_epoch),
                scheduler_specific_kwargs={"min_lr": self.config.trainer.inc_learning_rate * 0.01},
            )
        self.optimizer, self.scheduler = self.strategy.engine.prepare(self.optimizer, self.scheduler)

    def _freeze_before_increment_training(self, task_id):
        if task_id == 0:
            for name, param in self.model.named_parameters():
                if f'adapter' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for name, param in self.model.named_parameters():
                if f'adapter_{task_id}' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.model = self.strategy.engine.prepare(self.model)


if __name__ == "__main__":
    main()