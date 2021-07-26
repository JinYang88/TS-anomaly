import torch
from torch.utils.data import DataLoader
import numpy as np
from IPython import embed
from torch.utils.data import Dataset

from sklearn.preprocessing import MultiLabelBinarizer

test = pd.Series([["a", "b", "e"], ["c", "a"], ["d"], ["d"], ["e"]])

mlb = MultiLabelBinarizer()
res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)


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
        y = torch.zeros(26)
        y_indice = list(
            map(lambda x: self.vocab.label2idx[x], self.windows[idx, -1, :])
        )
        y[y_indice] = 1
        return {
            "x": self.windows[idx, :-1, :],
            "y": torch.LongTensor(y),
        }
