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
from pyts.approximation import SymbolicAggregateApproximation
from common.utils import load_hdf5, save_hdf5


class preprocessor:
    def __init__(self):
        self.vocab_size = None
        self.discretizer_list = defaultdict(list)

    def save(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        print("Saving preprocessor into {}".format(filepath))
        with open(filepath, "wb") as fw:
            pickle.dump(self.__dict__, fw)

    def load(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        print("Loading preprocessor from {}".format(filepath))
        with open(filepath, "rb") as fw:
            self.__dict__.update(pickle.load(fw))
    
    def symbolize(self, data_dict, n_bins=26, strategy="normal", nrows=None):
        def add_postfix(x):
            postfix = np.array([["_"+str(i)]*x.shape[0] for i in range(x.shape[1])]).T
            return np.char.add(x, postfix)

        print("Discarding constant dimensions.")
        constant_cols = []
        for idx, col in enumerate(data_dict["train"].T):
            if len(set(col)) == 1:
                constant_cols.append(idx)
        reserved_cols = [idx for idx in range(data_dict["train"].shape[1]) if idx not in constant_cols]
        data_dict["train"] = data_dict["train"][:, reserved_cols]
        data_dict["test"] = data_dict["test"][:, reserved_cols]

        print("Convert time series to symbolics.")
        sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
        train_sax = sax.fit_transform(data_dict["train"].T[:, :nrows])
        test_sax = sax.transform(data_dict["test"].T[:, :nrows])

        data_dict["train_tokens"] = add_postfix(train_sax.T)
        data_dict["test_tokens"] = add_postfix(test_sax.T)
        return data_dict

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

    def normalize(self, data_dict, method="minmax"):
        print("Normalizing data")
        # method: minmax, standard, robust
        normalized_dict = defaultdict(dict)

        # fit_transform using train
        if method == "minmax":
            est = MinMaxScaler(clip=True)
        elif method == "standard":
            est = StandardScaler()
        elif method == "robust":
            est = RobustScaler()

        train_ = est.fit_transform(data_dict["train"])
        test_ = est.transform(data_dict["test"])

        # assign back
        normalized_dict["train"] = train_
        normalized_dict["test"] = test_

        for k, v in data_dict.items():
            if k not in ["train", "test"]:
                normalized_dict[k] = v
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


def generate_windows_with_index(
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

    print("Generating sliding windows (size {}).".format(window_size))

    if "train" in data_dict:
        train = data_dict["train"][0:nrows]
        train_windows, _ = get_windows(train, window_size=window_size, stride=stride)

    if "test" in data_dict:
        test = data_dict["test"][0:nrows]
        test_label = (
            None
            if "test_labels" not in data_dict
            else data_dict["test_labels"][0:nrows]
        )
        test_windows, test_labels = get_windows(
            test, test_label, window_size=window_size, stride=1
        )

    if len(train_windows) > 0:
        results["train_windows"] = train_windows
        print("Train windows #: {}".format(train_windows.shape))

    if len(test_windows) > 0:
        if test_label is not None:
            results["test_windows"] = test_windows
            results["test_labels"] = test_labels
        else:
            results["test_windows"] = test_windows
        print("Test windows #: {}".format(test_windows.shape))

    idx = np.asarray(list(range(0, test.shape[0] + stride * window_size)))
    i = 0
    ts_len = test.shape[0]
    windows = []
    while i + window_size < ts_len:
        windows.append(idx[i : i + window_size])
        i += 1

    index = np.array(windows)

    results["index_windows"] = index

    # save_hdf5(cache_file, results)
    return results


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

    print("Generating sliding windows (size {}).".format(window_size))

    if "train" in data_dict:
        train = data_dict["train"][0:nrows]
        train_windows, _ = get_windows(train, window_size=window_size, stride=stride)

    if "test" in data_dict:
        test = data_dict["test"][0:nrows]
        test_label = (
            None
            if "test_labels" not in data_dict
            else data_dict["test_labels"][0:nrows]
        )
        test_windows, test_labels = get_windows(
            test, test_label, window_size=window_size, stride=1
        )

    if len(train_windows) > 0:
        results["train_windows"] = train_windows
        print("Train windows #: {}".format(train_windows.shape))

    if len(test_windows) > 0:
        if test_label is not None:
            results["test_windows"] = test_windows
            results["test_labels"] = test_labels
        else:
            results["test_windows"] = test_windows
        print("Test windows #: {}".format(test_windows.shape))

    # save_hdf5(cache_file, results)
    return results
