import torch
from torch.utils.data import Dataset
from torch import Tensor, LongTensor
import numpy as np

import csv
from os import walk

from ..utils.bert_utils import preprocessing_for_bert

__all__ = [
    "DatasetUniformNegatives",
    "DatasetValidation",
    "DatasetTest"
]


class DatasetUniformNegatives(Dataset):
    def __init__(self, pos_list, neg_list, neg_ratio, device, tokenizer):

        pos_tokenized = [tokenizer.encode(sent, add_special_tokens=True) for sent in pos_list]
        max_len = max([len(sent) for sent in pos_tokenized])
        negs_tokenized = [tokenizer.encode(sent, add_special_tokens=True) for sent in neg_list]
        max_len = max(max([len(sent) for sent in negs_tokenized]), max_len)

        print('Max length: ', max_len)

        print('Tokenizing data...')
        self.pos_input_ids, self.pos_masks = preprocessing_for_bert(pos_list, max_len, tokenizer)
        self.neg_input_ids, self.neg_masks = preprocessing_for_bert(neg_list, max_len, tokenizer)

        self.neg_ratio = neg_ratio
        self.to(device)

    def __len__(self):
        return self.pos_input_ids.shape[0]

    def __getitem__(self, idx: LongTensor):
        pos_sent_ids = self.pos_input_ids[idx]
        pos_masks = self.pos_masks[idx]

        # Generating Negatives
        neg_idxs = torch.randint(low=0, high=self.neg_input_ids.shape[0], size=(self.neg_ratio * pos_sent_ids.shape[0],),
                                 device=self.neg_input_ids.device)
        neg_sent_ids = self.neg_input_ids[neg_idxs]
        neg_masks = self.neg_masks[neg_idxs]

        return torch.cat((pos_sent_ids, neg_sent_ids)), torch.cat((pos_masks, neg_masks)), idx.shape[0]

    def to(self, device):
        self.pos_input_ids = self.pos_input_ids.to(device)
        self.pos_masks = self.pos_masks.to(device)
        self.neg_input_ids = self.neg_input_ids.to(device)
        self.neg_masks = self.neg_masks.to(device)

    @classmethod
    def from_csv(cls, train_path: str, neg_ratio: int, device, tokenizer):
        pos_list = []
        neg_list = []
        print("Loading Train CSV")
        with open(train_path, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            skip_header = True
            for row in csv_reader:
                if skip_header:
                    skip_header = False
                    continue
                if row[1] == "1":
                    pos_list.append(row[0])
                else:
                    neg_list.append(row[0])
        return cls(pos_list, neg_list, neg_ratio, device, tokenizer)


class DatasetValidation(object):
    def __init__(self, data, labels, tokenizer, device):

        self.labels = np.array(labels, dtype=np.int64)
        self.val_data = data

        data_tokenized = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
        max_len = max([len(sent) for sent in data_tokenized])

        print('Max length: ', max_len)

        print('Tokenizing data...')
        self.val_sent_ids, self.val_masks = preprocessing_for_bert(data, max_len, tokenizer)

        self.to(device)

    def to(self, device):
        self.val_sent_ids = self.val_sent_ids.to(device)
        self.val_masks = self.val_masks.to(device)

    @classmethod
    def from_csv(cls, val_dir: str, tokenizer, device):
        data = []
        labels = []
        val_files = next(walk(val_dir), (None, None, []))[2]
        print("Loading Validation CSV")
        with open(val_dir + val_files[0], newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            skip_header = True
            for row in csv_reader:
                if skip_header:
                    skip_header = False
                    continue
                data.append(row[0])
                labels.append(row[1])
        return cls(data, labels, tokenizer, device)


class DatasetTest(object):
    def __init__(self, test_data, test_labels, tokenizer):
        self.test_sent_ids = []
        self.test_labels = test_labels
        self.test_masks = []
        self.test_data = test_data
        for data in test_data:
            data_tokenized = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
            max_len = max([len(sent) for sent in data_tokenized])

            print('Max length: ', max_len)

            print('Tokenizing data...')
            data_sent_ids, data_masks = preprocessing_for_bert(data, max_len, tokenizer)
            self.test_sent_ids.append(data_sent_ids)
            self.test_masks.append(data_masks)

    @classmethod
    def from_csv(cls, test_dir: str, tokenizer):
        test_data = []
        test_labels = []

        test_files = next(walk(test_dir), (None, None, []))[2]
        print("Loading Test CSV")
        for test_file in test_files:
            data = []
            labels = []
            with open(test_dir + test_file, newline='') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                skip_header = True
                for row in csv_reader:
                    if skip_header:
                        skip_header = False
                        continue
                    data.append(row[0])
                    labels.append(row[1])
            test_data.append(data.copy())
            test_labels.append(labels.copy())
        return cls(test_data, test_labels, tokenizer)
