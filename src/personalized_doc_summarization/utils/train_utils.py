from pathlib import Path
from typing import Optional, Union
import torch
from torch.nn import Module

from ..utils.exceptions import EarlyStoppingException

import warnings


def cuda_if_available(use_cuda: Optional[bool] = None) -> torch.device:
    cuda_available = torch.cuda.is_available()
    _use_cuda = (use_cuda is None or use_cuda) and cuda_available
    if use_cuda is True and not cuda_available:
        warnings.warn("Requested CUDA but it is not available, running on CPU")
    if use_cuda is False and cuda_available:
        warnings.warn(
            "Running on CPU, even though CUDA is available. "
            "(This is likely not desired, check your arguments.)"
        )
    return torch.device("cuda" if _use_cuda else "cpu")


class EarlyStopping:
    """
    Stop looping if a value is stagnant.
    """

    def __init__(self,
                 name: str = "EarlyStopping Value",
                 patience: int = 10):
        self.name = name
        self.patience = patience
        self.count = 0
        self.value = None

    def __call__(self, value) -> None:
        if value == self.value:
            self.count += 1
            if self.count >= self.patience:
                raise EarlyStoppingException(
                    f"{self.name} has not changed in {self.patience} steps."
                )
        else:
            self.value = value
            self.count = 0


class ModelCheckpoint:
    """
    When called, saves current model parameters to system RAM.
    """

    def __init__(self,
                 run_dir: Path,
                 filename: str = "trained_model.pt"):
        self.run_dir = run_dir
        self.filename = filename

    def path(self):
        return self.run_dir / self.filename

    def __call__(self, model: Module) -> None:
        self.best_model_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    def save_to_disk(self, _) -> None:
        """
        Saves model which was previously saved to RAM to disk.

        :param _: throw-away parameter (needed to fit the TrainLooper calling style)
        :return: None
        """
        print(f"Saving model as '{self.path}'...", end="", flush=True)
        torch.save(self.best_model_state_dict, self.path)
        print("done!")


class IntervalConditional:
    def __init__(self,
                 interval: Union[int, float],
                 last: Union[int, float] = 0):
        self.interval = interval
        self.last = last

    def __call__(self, value: Union[int, float]) -> bool:
        """Return True when value has exceeded the previous True by at least self.interval, otherwise return False"""
        if value - self.last >= self.interval:
            self.last = value
            return True
        else:
            return False

    @classmethod
    def interval_conditional_converter(cls, value):
        """Converter helper function for IntervalConditional"""
        if value is None:
            return None
        else:
            return IntervalConditional(value)
