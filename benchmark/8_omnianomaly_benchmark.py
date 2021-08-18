# -*- coding: utf-8 -*-
import sys

sys.path.append("../")
import os
import warnings

warnings.filterwarnings("ignore")

import traceback
import hashlib
import logging
import pickle
import time
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tfsnippet.examples.utils import MLResults
from common.dataloader import load_dataset, get_data_dim
from common.config import subdatasets
from common.data_preprocess import generate_windows, preprocessor
from networks.omni_anomaly.detector import OmniDetector, DataGenerator
from common.evaluation import evaluator
from tfsnippet.utils import Config
from tensorflow.python.keras.utils import Sequence
from common.utils import pprint
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

#  python 8_omnianomaly_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 32 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 1

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")
parser.add_argument("--z_dim", type=int, help="z_dim")
parser.add_argument("--rnn_num_hidden", type=int, help="rnn_num_hidden")
parser.add_argument("--window_size", type=int, help="window_size")
parser.add_argument("--dense_dim", type=int, help="dense_dim")
parser.add_argument("--nf_layers", type=int, help="nf_layers")
parser.add_argument("--max_epoch", type=int, help="max_epoch")

parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--stride", type=int, help="stride")

parser.add_argument("--gpu", type=int, default=0, help="The gpu index, -1 for cpu")

args = vars(parser.parse_args())

model_name = "omnianomaly"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]
dataset = args["dataset"]


class ExpConfig(Config):
    dataset = args["dataset"]

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = args["z_dim"]
    rnn_cell = "GRU"  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = args["rnn_num_hidden"]
    window_length = args["window_size"]
    dense_dim = args["dense_dim"]
    posterior_flow_type = "nf"  # 'nf' or None
    nf_layers = args["nf_layers"]  # for nf
    max_epoch = args["max_epoch"]
    stride = args["stride"]
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 1024
    l2_reg = 0.0001
    initial_lr = args["lr"]
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 2048
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

    level = 0.005

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    restore_dir = None  # If not None, restore variables from this dir
    save_dir = "model"
    result_dir = "result"  # Where to save the result file
    train_score_filename = "train_score.pkl"
    test_score_filename = "test_score.pkl"


if __name__ == "__main__":
    for subdataset in subdatasets[dataset][0:1]:
        try:
            print(f"Running on {subdataset} of {dataset}")
            config = ExpConfig()
            config.x_dim = get_data_dim(dataset)

            save_path = os.path.join("./savd_dir_omni", hash_id, subdataset)
            config.result_dir = os.path.join(save_path, "result")
            config.save_dir = os.path.join(save_path, "model")

            results = MLResults(config.result_dir)
            results.save_config(config)  # save experiment settings for review
            results.make_dirs(config.save_dir, exist_ok=True)

            data_dict = load_dataset(dataset, subdataset, "all", root_dir="../")

            # preprocessing
            pp = preprocessor()
            data_dict = pp.normalize(data_dict)

            # generate sliding windows
            window_dict = generate_windows(
                data_dict, window_size=config.window_length, stride=config.stride
            )

            # batch data
            x_train = DataGenerator(window_dict["train_windows"])
            x_test = DataGenerator(window_dict["test_windows"])
            test_labels = DataGenerator(window_dict["test_labels"])

            od = OmniDetector(config)
            od.fit(x_train)

            anomaly_score = od.predict_prob(x_test)
            anomaly_score_train = od.predict_prob(x_train)
            anomaly_label = window_dict["test_labels"][
                :, -1
            ]  # last point of each window
            print(anomaly_score.shape, anomaly_label.shape)

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                {"train": anomaly_score_train, "test": anomaly_score},
                anomaly_label,
                od.time_tracker,
            )
        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )