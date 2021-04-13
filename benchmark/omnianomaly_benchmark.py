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


def minibatch_slices_iterator(length, batch_size, ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.
    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)
    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.
    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.
    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(
        self,
        array_size,
        window_size,
        batch_size,
        excludes=None,
        shuffle=False,
        ignore_incomplete_batch=False,
    ):
        # check the parameters
        if window_size < 1:
            raise ValueError("`window_size` must be at least 1")
        if array_size < window_size:
            raise ValueError(
                "`array_size` must be at least as large as " "`window_size`"
            )
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError(
                    "The shape of `excludes` is expected to be "
                    "{}, but got {}".format(expected_shape, excludes.shape)
                )

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.
        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.
        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.
        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError("`arrays` must not be empty")

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
            length=len(self._indices),
            batch_size=self._batch_size,
            ignore_incomplete_batch=self._ignore_incomplete_batch,
        ):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)[0]


#  python omnianomaly_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 32 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10

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
    save_dir = "model"
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = "result"  # Where to save the result file
    train_score_filename = "train_score.pkl"
    test_score_filename = "test_score.pkl"


if __name__ == "__main__":
    for subdataset in subdatasets[dataset][0:1]:
        try:
            print(f"Running on {subdataset} of {dataset}")
            config = ExpConfig()
            config.x_dim = get_data_dim(dataset)

            results = MLResults(config.result_dir)
            results.save_config(config)  # save experiment settings for review
            results.make_dirs(config.save_dir, exist_ok=True)

            data_dict = load_dataset(dataset, subdataset, nrows=None)

            # preprocessing
            # pp = preprocessor()
            # data_dict = pp.normalize(data_dict)

            x_train = list(
                BatchSlidingWindow(
                    array_size=len(data_dict["train"]),
                    window_size=config.window_length,
                    batch_size=config.batch_size,
                ).get_iterator([data_dict["train"]])
            )
            x_test = list(
                BatchSlidingWindow(
                    array_size=len(data_dict["test"]),
                    window_size=config.window_length,
                    batch_size=config.batch_size,
                ).get_iterator([data_dict["test"]])
            )
            test_labels = data_dict["test_labels"]
            # # generate sliding windows
            # window_dict = generate_windows(
            #     data_dict, window_size=config.window_length, stride=config.stride
            # )

            # # batch data
            # x_train = DataGenerator(window_dict["train_windows"], shuffle=True)
            # x_test = DataGenerator(window_dict["test_windows"], shuffle=False)
            # test_labels = DataGenerator(window_dict["test_labels"], shuffle=False)

            od = OmniDetector(config)
            anomaly_score = od.fit(x_train, x_test)

            # anomaly_score = od.predict_prob(x_test)
            anomaly_label = data_dict["test_labels"][
                -len(anomaly_score) :
            ]  # last point of each window
            print(anomaly_score.shape, anomaly_label.shape)

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                anomaly_score,
                anomaly_label,
                od.time_tracker,
            )
        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )