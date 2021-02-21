import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from IPython import embed
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from common.sliding import BatchSlidingWindow
from common.utils import load_hdf5, save_hdf5


class preprocessor:
    def __init__(self):
        self.vocab_size = None
        self.discretizer_list = defaultdict(list)

    def save(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        logging.info("Saving preprocessor into {}".format(filepath))
        with open(filepath, "wb") as fw:
            pickle.dump(self.__dict__, fw)

    def load(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        logging.info("Loading preprocessor from {}".format(filepath))
        with open(filepath, "rb") as fw:
            self.__dict__.update(pickle.load(fw))

    def discretize(self, data_dict, n_bins=1000):
        if n_bins is None:
            n_bins = 50
        discretized_dict = defaultdict(dict)
        for data_name, arr in data_dict.items():
            if not data_name in ["train", "test"]:
                discretized_dict[data_name] = arr
                continue
            if data_name == "test":
                test_ = self.discretizer.transform(arr)
                discretized_dict[data_name] = test_.astype(int)
            elif data_name == "train":
                # fit_transform using train
                self.discretizer = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="uniform"
                )
                train_ = self.discretizer.fit_transform(arr)
                discretized_dict[data_name] = train_.astype(int)
        return discretized_dict

    def build_vocab(self, data_dict):
        max_index = -float("inf")
        index = np.max(data_dict["train"].reshape(-1))
        self.vocab_size = index + 1
        logging.info("# of Discretized tokens: {}".format(self.vocab_size))

        return self.vocab_size

    def normalize(self, data_dict, method="standard"):
        logging.info("Normalizing data")
        # method: minmax, standard, robust
        normalized_dict = defaultdict(dict)
        for data_name, sub_dict in data_dict.items():
            if not isinstance(sub_dict, dict):
                normalized_dict[data_name] = sub_dict
                continue

            # fit_transform using train
            train = sub_dict["train"]  # shape: time x dim
            if method == "minmax":
                est = MinMaxScaler()
            elif method == "standard":
                est = StandardScaler()
            elif method == "robust":
                est = RobustScaler()

            train_ = est.fit_transform(train)

            # transform test
            test = sub_dict["test"]
            test_ = est.transform(test)

            # assign back
            normalized_dict[data_name]["train"] = train_
            normalized_dict[data_name]["test"] = test_

            for k, v in sub_dict.items():
                if k not in ["train", "test"]:
                    normalized_dict[data_name][k] = v
        return normalized_dict


def get_windows(ts, labels=None, window_size=128, stride=1, dim=None):
    i = 0
    ts_len = ts.shape[0]
    windows = []
    label_windows = []
    while i + window_size < ts_len:
        if dim is not None:
            windows.append(ts[i : i + window_size, dim])
        else:
            windows.append(ts[i : i + window_size])
        if labels is not None:
            label_windows.append(labels[i : i + window_size])
        i += stride
    if labels is not None:
        return np.array(windows, dtype=np.float32), np.array(
            label_windows, dtype=np.float32
        )
    else:
        return np.array(windows, dtype=np.float32), None


def generate_windows(
    data_dict,
    data_hdf5_path=None,
    window_size=100,
    nrows=None,
    clear=False,
    stride=1,
    **kwargs
):
    results = {}

    if data_hdf5_path:
        cache_file = os.path.join(
            data_hdf5_path,
            "hdf5",
            "window_dict_ws={}_st={}_nrows={}.hdf5".format(window_size, stride, nrows),
        )
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if not clear and os.path.isfile(cache_file):
            return load_hdf5(cache_file)

    logging.info("Generating sliding windows (size {}).".format(window_size))

    if "train" in data_dict:
        train = data_dict["train"][0:nrows]
        train_windows, _ = get_windows(train, window_size=window_size, stride=stride)

    if "test" in data_dict:
        test = data_dict["test"][0:nrows]
        test_label = (
            None if "test_label" not in data_dict else data_dict["test_label"][0:nrows]
        )
        test_windows, test_labels = get_windows(
            test, test_label, window_size=window_size, stride=1
        )

    if len(train_windows) > 0:
        results["train_windows"] = train_windows
        logging.info("Train windows #: {}".format(train_windows.shape))

    if len(test_windows) > 0:
        if test_label is not None:
            results["test_windows"] = test_windows
            results["test_labels"] = test_labels
        else:
            results["test_windows"] = test_windows
        logging.info("Test windows #: {}".format(test_windows.shape))

    # save_hdf5(cache_file, results)
    return results
