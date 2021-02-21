import torch
import numpy as np
from IPython import embed
from torch.utils.data import DataLoader


class WindowIterator:
    def __init__(self, windows, batch_size, shuffle, num_workers=2):
        self.windows = windows
        self.loader = DataLoader(
            windows, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def fetch_windows(self):
        return self.windows
