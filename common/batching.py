import torch
from torch.utils.data import DataLoader
import numpy as np
from IPython import embed
from torch.utils.data import Dataset

from sklearn.preprocessing import MultiLabelBinarizer


class WindowIterator:
    def __init__(self, windows, batch_size, shuffle, num_workers=2):
        self.windows = windows
        self.loader = DataLoader(
            windows, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


class TokenDataset(Dataset):
    def __init__(self, vocab, windows_tokens, windows, nb_classes, batch_size, shuffle, num_workers=2):
        self.vocab = vocab
        self.windows = windows
        self.windows_tokens = windows_tokens
        self.nb_classes = nb_classes
        self.loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        y = torch.zeros(self.nb_classes)
        y_indice = list(
            map(lambda x: self.vocab.label2idx[x], self.windows_tokens[idx, -1, :])
        )
        y[y_indice] = 1
        return {
            "x": self.windows_tokens[idx, :-1, :],
            "y": y.float(),
        }
        # y = torch.LongTensor(list(
        #     map(lambda x: self.vocab.label2idx[x], self.windows_tokens[idx, -1, :])
        # ))
        # return {
        #     "x": self.windows_tokens[idx, :-1, :],
        #     "y": y,
        # }
