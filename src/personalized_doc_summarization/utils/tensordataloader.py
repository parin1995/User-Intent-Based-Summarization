import math
from torch.utils.data import Sampler, TensorDataset, DataLoader
import torch

__all__ = [
    "TensorBatchSampler",
    "TensorDataLoader",
]


######################################
# Custom Sampler
######################################


class TensorBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=False, drop_last=False):
        # We don't need to check for this, other datasets might support this type of
        # list-based indexing.
        # if not isinstance(data_source, TensorDataset):
        #     raise ValueError(
        #         f"data_source should be an instance of torch.utils.data.TensorDataset, but got data_source={data_source}"
        #     )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(shuffle, bool):
            raise ValueError(
                f"shuffle should be a boolean value, but got shuffle={shuffle}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(len(self.data_source))
        else:
            self.idxs = torch.arange(len(self.data_source))
        self.current_idx = 0
        self.next_idx = self.batch_size
        return self

    def __next__(self):
        out = self.idxs[self.current_idx : self.next_idx]
        out_of_data = self.current_idx >= len(self.data_source)
        not_full_batch = self.next_idx > len(self.data_source)
        if out_of_data or (not_full_batch and self.drop_last):
            del self.idxs, self.current_idx, self.next_idx
            raise StopIteration
        else:
            self.current_idx = self.next_idx
            self.next_idx += self.batch_size
            return out

    def __len__(self):
        return (math.floor if self.drop_last else math.ceil)(
            len(self.data_source) / self.batch_size
        )


def _unwrap_collate_fn(batch):
    return batch[0]


class TensorDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        *,
        drop_last=False,
        collate_fn=None,
        **kwargs,
    ):
        if sampler is not None or batch_sampler is not None or collate_fn is not None:
            raise ValueError(
                "TensorDataLoader does not support alternate samplers, batch samplers, or collate functions."
            )
        sampler = TensorBatchSampler(dataset, batch_size, shuffle, drop_last)
        super().__init__(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            collate_fn=_unwrap_collate_fn,
            **kwargs,
        )
