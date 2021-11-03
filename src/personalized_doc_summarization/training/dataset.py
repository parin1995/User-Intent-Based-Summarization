import torch
from torch.utils.data import Dataset
from torch import Tensor, LongTensor


class DocumentDatasetNegativeSampling(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: LongTensor):
        pass

    def to(self, device):
        pass

    @classmethod
    def from_csv(cls, path: str, *args, **kwargs):
        return cls(*args, **kwargs)