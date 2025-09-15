from abc import ABC, abstractmethod
from accelerate import Accelerator
import torch.optim as optim
from torchdata.stateful_dataloader import StatefulDataLoader
import torch
import os
import shutil
import logging

class BaseStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.engine: Accelerator = None

    @abstractmethod
    def print(self, *msg):
        raise NotImplementedError

    @abstractmethod
    def setup_distributed(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_rank0(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_optimizer(self, model):
        raise NotImplementedError

    @abstractmethod
    def setup_dataloader(self, dataset,pin_memory: bool = False,shuffle=True, drop_last=True):
        raise NotImplementedError

    @abstractmethod
    def all_reduce(self, data, op="mean"):
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, model, tag, save_dir, max_num=3, max_mem=1000, client_states=None):
        raise NotImplementedError

    @abstractmethod
    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
       raise NotImplementedError



class AccelerateStrategy(BaseStrategy):
    """
    The strategy for training with Accelerate.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def setup_distributed(self):
        self.engine = Accelerator(**self.config.strategy)

    def create_optimizer(self, model):
        if self.config.optim.type == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.trainer.learning_rate,
                betas = self.config.optim.betas,
                weight_decay=self.config.optim.weight_decay,
            )
            return optimizer
        else:
            raise NotImplementedError()

    def setup_dataloader(self, dataset,pin_memory: bool = False, shuffle=True, collate_fn=None):
        return StatefulDataLoader(
            dataset,
            batch_size=self.config.trainer.train_batch_size,
            drop_last=self.config.data.drop_last,
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def print(self, *msg):
        self.engine.print(*msg)

    def is_rank0(self) -> bool:
        return self.engine.process_index == 0

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")

        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                if torch.cuda.is_available():
                    data = data.to(torch.cuda.current_device())
                else:
                    # No CUDA available, CPU tensor will not be moved to GPU for all_reduce.
                    return data.item()

            data = self.engine.reduce(data, op)
            if is_cpu_tensor:
                data = data.cpu()

            return data.item() if not is_tensor else data

    def save_ckpt(self, model, tag, save_dir, max_num=3, max_mem=1000, client_states=None):
        ckpt_dir = str(os.path.join(save_dir, tag))
        if client_states is None:
            client_states = {}
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        with open(os.path.join(save_dir, f'chosen_ckpt.txt'), 'w') as f:
            f.write(tag)
        if self.is_rank0():
            os.makedirs(save_dir, exist_ok=True)
            MAX_SIZE = max_mem * 1024**3  # Convert GB to bytes

            while True:
                subdirs = sorted(
                    [
                        (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) > max_num or total_size > MAX_SIZE:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

            # save client states
            import pickle
            with open(os.path.join(ckpt_dir, 'client_states.pkl'), "wb") as f:
                pickle.dump(client_states, f) # type: ignore[arg-type]

        # TODO: we may need to consider hf models in the future
        # save local model's ckpt
        self.engine.save_state(output_dir=ckpt_dir)

        self.print(f"Saved ckpt to {ckpt_dir}")

        # Explicitly release memory
        import gc

        gc.collect()

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        with open(os.path.join(load_dir, f'chosen_ckpt.txt'), 'r') as f:
            tag = f.readline() if tag is None else tag
        assert tag is not None, "ckpt path should exist"
        ckpt_dir = os.path.join(load_dir, tag)
        # TODO: Somehow we need to set weights_only=False, related issue:
        # https://github.com/huggingface/accelerate/issues/3539
        self.engine.load_state(ckpt_dir, load_kwargs={'weights_only': False}) # type: ignore[arg-type]

        import pickle
        with open(os.path.join(ckpt_dir, 'client_states.pkl'), "rb") as f: # type: ignore[arg-type]
            client_states = pickle.load(f)

        # To alight the implementation of load_ckpt in Deepspeed strategy
        return None, client_states

# class DeepspeedStrategy(BaseStrategy):
#     """
#     The strategy for training with Deepspeed.
#     """
#     def __init__(self, config) -> None:
#         super().__init__(config)
#
#     def setup_distributed(self):
#         raise NotImplementedError