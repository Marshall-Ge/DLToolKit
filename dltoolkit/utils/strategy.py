from abc import ABC

from torch import distributed as dist
from accelerate import Accelerator

class BaseStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.engine = None

    def is_rank_0(self) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def setup_distributed(self) -> None:
        raise NotImplementedError()



class AccelerateStrategy(BaseStrategy):
    """
    The strategy for training with Accelerate.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def setup_distributed(self):
        self.engine = Accelerator(**self.config.strategy)


class DeepspeedStrategy(BaseStrategy):
    """
    The strategy for training with Deepspeed.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def setup_distributed(self):
        raise NotImplementedError()