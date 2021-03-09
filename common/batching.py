import torch
from torch.utils.data import DataLoader
import numpy as np
from IPython import embed


class WindowIterator:
    def __init__(self, windows, batch_size, shuffle, num_workers=2):
        self.windows = windows
        self.loader = DataLoader(
            windows, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
