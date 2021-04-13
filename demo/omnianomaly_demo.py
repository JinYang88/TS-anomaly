# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("../")

import logging
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tfsnippet.examples.utils import MLResults
from common.dataloader import load_dataset, get_data_dim
from common.data_preprocess import generate_windows, preprocessor
from networks.omni_anomaly.detector import OmniDetector
from IPython import embed
from common.evaluation import evaluator
from tfsnippet.utils import Config
from tensorflow.python.keras.utils import Sequence
from common.utils import pprint

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


dataset = "SMD"
subdataset = "machine-1-1"
point_adjustment = True
iterate_threshold = True


class DataGenerator(Sequence):
    def __init__(
        self,
        data_array,
        batch_size=32,
        shuffle=False,
    ):
        self.darray = data_array
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_pool = list(range(self.darray.shape[0]))
        self.length = int(np.ceil(len(self.index_pool) * 1.0 / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        indexes = self.index_pool[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X = self.darray[indexes]

        # in case on_epoch_end not be called automatically :)
        if index == self.length - 1:
            self.on_epoch_end()
        return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_pool)


class ExpConfig(Config):
    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = "GRU"  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = "nf"  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 1
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 256
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.0
    bf_search_max = 400.0
    bf_search_step_size = 1.0

    valid_step_freq = 100
    gradient_clip_norm = 10.0

    early_stop = False  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.07

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = "model"
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = "result"  # Where to save the result file
    train_score_filename = "train_score.pkl"
    test_score_filename = "test_score.pkl"


if __name__ == "__main__":
    config = ExpConfig()
    config.x_dim = get_data_dim(dataset)

    # print_with_title("Configurations", pformat(config.to_dict()), after="\n")
    # open the result object and prepare for result directories if specified
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs(config.save_dir, exist_ok=True)

    logging.basicConfig(
        level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    data_dict = load_dataset(dataset, subdataset)

    # preprocessing
    pp = preprocessor()
    data_dict = pp.normalize(data_dict)

    # generate sliding windows
    window_dict = generate_windows(
        data_dict, window_size=config.window_length, stride=5
    )

    # batch data
    x_train = DataGenerator(window_dict["train_windows"])
    x_test = DataGenerator(window_dict["test_windows"])
    test_labels = DataGenerator(window_dict["test_labels"])

    od = OmniDetector(config)
    anomaly_score = od.fit(x_train, x_test)

    # anomaly_score = od.predict_prob(x_test)
    anomaly_label = window_dict["test_labels"][:, -1]  # last point of each window
    print(anomaly_score.shape, anomaly_label.shape)
    # Make evaluation
    eva = evaluator(
        ["auc", "f1", "pc", "rc"],
        anomaly_score,
        anomaly_label,
        iterate_threshold=iterate_threshold,
        iterate_metric="f1",
        point_adjustment=point_adjustment,
    )
    eval_results = eva.compute_metrics()

    pprint(eval_results)