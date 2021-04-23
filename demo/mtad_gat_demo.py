import json
import logging
import math
import os

import nni
import numpy
import pandas
import torch
import pandas as pd
from IPython import embed

from common import data_preprocess
from common.config import (
    initialize_config,
    parse_arguments,
    set_logger,
    subdatasets,
    get_trial_id,
)
from common.dataloader import load_dataset
from common.sliding import WindowIterator
from common.utils import print_to_json, update_from_nni_params, seed_everything

seed_everything(2020)

dataset = "SMD"
subdataset = "machine-1-1"
window_size = 32
stride = 5
intermediate_dim = 64
z_dim = 3
stateful = True
learning_rate = 0.001
batch_size = 64
test_batch_size = 1024
num_epochs = 1
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load & preprocess data
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    # preprocessing
    pp = preprocessor()
    data_dict = pp.normalize(data_dict)

    print(data_dict)

    # generate sliding windows
    window_dict = generate_windows(data_dict, window_size=window_size, stride=stride)

    train = window_dict["train_windows"]
    test = window_dict["test_windows"]
    test_labels = window_dict["test_labels"]

    train_iterator = WindowIterator(
        window_dict["train_windows"], batch_size=batch_size, shuffle=True
    )
    test_iterator = WindowIterator(
        window_dict["test_windows"], batch_size=test_batch_size, shuffle=False
    )
