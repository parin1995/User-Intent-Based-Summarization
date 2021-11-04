import torch
import wandb

from torch.nn import Module, Linear, ReLU, Sigmoid
from torch import Tensor

__all__ = [
    "FineTuneBert"
]


class FineTuneBert(Module):
    def __init__(self, bert_model, pretrained_weights, input_dim: int, hidden_dim: int = None,
                 output_dim: int = 1, freeze_bert: bool = True):

        super().__init__()
        self.bert = bert_model.from_pretrained(pretrained_weights)
        self.fc1 = Linear(input_dim, output_dim)
        # self.relu1 = ReLU()
        # self.fc2 = Linear(hidden_dim, output_dim)
        self.sigmoid = Sigmoid()

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask):

        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        out = self.fc1(last_hidden_state_cls)
        out = self.sigmoid(out)
        return out
