import torch
from torch.utils.data import DataLoader
import numpy as np
from IPython import embed
from torch.utils.data import Dataset


class WindowIterator:
    def __init__(self, windows, batch_size, shuffle, num_workers=2):
        self.windows = windows
        self.loader = DataLoader(
            windows, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


class TokenDataset(Dataset):
    def __init__(self, vocab, windows, batch_size, shuffle, num_workers=2):
        self.vocab = vocab
        self.windows = windows
        self.loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            "x": self.windows[idx, :-1, :],
            "y": torch.LongTensor(
                list(map(lambda x: self.vocab.label2idx[x], self.windows[idx, -1, :]))
            ),
        }
