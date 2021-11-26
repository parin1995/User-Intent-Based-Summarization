import torch
import wandb
from torch import Tensor
from torch.nn import Module

__all__ = [
    "BCEWithWeightedLoss"
]


class BCEWithWeightedLoss(Module):
    def __init__(self, positive_weight: float):
        super().__init__()
        self.pos_weight = positive_weight

    def forward(self, probs: Tensor, num_pos: int = None, labels: Tensor = None, eval: bool = False):
        """
                pos_probs: Tensor => (num_pos,)
                neg_probs: Tensor => (num_pos*neg_ratio,)
        """
        if not eval:
            pos_probs = probs[:num_pos]
            neg_probs = probs[num_pos + 1:]
            # TODO: Use the formula discussed in the ipad
            # This is to calculate the training loss
        else:
            # TODO: Use the Normal Binary Cross Entropy Formula (But Weighted as in the if condition)
            #  with the labels parameter and the probs argument
            # This is to calculate the Validation loss
            pass


class BCELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: Tensor, num_pos: int = None, labels: Tensor = None, eval: bool = False):
        """
                pos_probs: Tensor => (num_pos,)
                neg_probs: Tensor => (num_pos*neg_ratio,)
        """
        if not eval:
            pos_probs = probs[:num_pos]
            neg_probs = probs[num_pos + 1:]
            pos_loss = -torch.log(pos_probs)
            neg_loss = -torch.log(1 - neg_probs)
            return torch.sum(pos_loss) + torch.sum(neg_loss)
        else:
            # This is to calculate the Validation loss
            pos_idxs = torch.nonzero(labels == 1).flatten()
            neg_idxs = torch.nonzero((labels == 0)).flatten()
            pos_loss = -torch.log(probs[pos_idxs])
            neg_loss = -torch.log(probs[neg_idxs])
            return torch.sum(pos_loss) + torch.sum(neg_loss)