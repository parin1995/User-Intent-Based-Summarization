import torch
import wandb
from torch.nn import Module
from torch import Tensor

__all__ = [
    "FineTuneBert"
]


class FineTuneBert(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_samples: Tensor, neg_samples: Tensor):
        pass
