from abc import ABC, abstractmethod
from accelerate import Accelerator
import torch.optim as optim
from torchdata.stateful_dataloader import StatefulDataLoader
import torch
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

    def setup_dataloader(self, dataset,pin_memory: bool = False, shuffle=True, collate_fn=None, drop_last=False):
        return StatefulDataLoader(
            dataset,
            batch_size=self.config.trainer.train_batch_size,
            drop_last=drop_last,
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


# class DeepspeedStrategy(BaseStrategy):
#     """
#     The strategy for training with Deepspeed.
#     """
#     def __init__(self, config) -> None:
#         super().__init__(config)
#
#     def setup_distributed(self):
#         raise NotImplementedError