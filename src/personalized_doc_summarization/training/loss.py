import torch
import wandb
from torch import Tensor
from torch.nn import Module

__all__ =[
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
            neg_probs = probs[num_pos+1:]
            # TODO: Use the formula discussed in the ipad
            # This is to calculate the training loss
        else:
            # TODO: Use the Normal Binary Cross Entropy Formula (But Weighted as in the if condition)
            #  with the labels parameter and the probs argument
            # This is to calculate the Validation loss
            pass


# Example Here:
class BCEWithLogsNegativeSamplingLoss(Module):
    def __init__(self, negative_weight: float = 0.5):
        super().__init__()
        self.negative_weight = negative_weight

    def forward(self, log_prob_scores: Tensor) -> Tensor:
        """
        Returns a weighted BCE loss where:
            (1 - negative_weight) * pos_loss + negative_weight * weighted_average(neg_loss)

        :param log_prob_scores: Tensor of shape (..., 1+K) where [...,0] is the score for positive examples and [..., 1:] are negative
        :return: weighted BCE loss
        """

        log_prob_pos = log_prob_scores[..., 0]
        log_prob_neg = log_prob_scores[..., 1:]
        pos_loss = -log_prob_pos
        neg_loss = -log1mexp(log_prob_neg)
        logit_prob_neg = log_prob_neg + neg_loss
        weights = F.softmax(logit_prob_neg, dim=-1)
        weighted_average_neg_loss = (weights * neg_loss).sum(dim=-1)
        final_loss = (
            1 - self.negative_weight
        ) * pos_loss + self.negative_weight * weighted_average_neg_loss

        return final_loss