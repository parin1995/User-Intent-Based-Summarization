import torch
from torch.utils.data import Dataset
from torch import Tensor, LongTensor
import csv

__all__ = [
    "DatasetUniformNegatives",
    "DocumentTestSet"
]


class DatasetUniformNegatives(Dataset):
    def __init__(self, pos_list, neg_list, neg_ratio, device, tokenizer):
        self.tokenizer = tokenizer

        pos_tokenized = pos_list.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        max_len = max([len(sent) for sent in pos_tokenized.values])
        negs_tokenized = neg_list.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        max_len = max(max([len(sent) for sent in negs_tokenized.values]), max_len)

        print('Max length: ', max_len)

        print('Tokenizing data...')
        self.pos_input_ids, self.pos_masks = self.preprocessing_for_bert(pos_list)
        self.neg_input_ids, self.neg_masks = self.preprocessing_for_bert(neg_list)

        self.neg_ratio = neg_ratio
        self.to(device)

    def __len__(self):
        return self.pos_input_ids.shape[0]

    def __getitem__(self, idx: LongTensor):
        pos_sent_ids = self.pos_input_ids[idx]
        pos_masks = self.pos_masks[idx]

        # Generating Negatives
        neg_idxs = torch.randint(low=0, high=self.neg.shape[0], size=(self.neg_ratio * pos_sent_ids.shape[0],), device=self.neg.device)
        neg_sent_ids = self.neg_input_ids[neg_idxs]
        neg_masks = self.neg_masks[neg_idxs]

        return torch.cat((pos_sent_ids, neg_sent_ids)), torch.cat((pos_masks, neg_masks)), idx.shape[0]

    def to(self, device):
        self.pos = self.pos.to(device)
        self.neg = self.neg.to(device)

    def preprocessing_for_bert(self, data, max_length):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max_length,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    @classmethod
    def from_csv(cls, train_path: str, neg_ratio: int, device, tokenizer):
        pos_list = []
        neg_list = []
        print("Loading CSV")
        with open(train_path, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            skip_header = True
            for row in csv_reader:
                if skip_header:
                    skip_header = False
                    continue
                row_list = row.split(",")
                if row_list[1] == "1":
                    pos_list.append(row_list[0])
                else:
                    neg_list.append(row_list[1])
        return cls(pos_list, neg_list, neg_ratio, tokenizer)


class DocumentTestSet(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: LongTensor):
        pass

    def to(self, device):
        pass

    @classmethod
    def from_csv(cls, test_dir: str, *args, **kwargs):

        return cls(*args, **kwargs)