# -*- coding: utf-8 -*-
import os
import pickle
import copy

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from glob import glob
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

prefix = "processed"

data_path_dict = {
    "SMD": "./datasets/anomaly/SMD/processed",
    "SMAP": "./datasets/anomaly/SMAP-MSL/processed_SMAP",
    "MSL": "./datasets/anomaly/SMAP-MSL/processed_MSL",
}


def get_data_dim(dataset):
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif dataset == "SMD" or str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def load_dataset(dataset, subdataset, use_dim="all", nrows=None):
    print("Loading {} of {} dataset".format(subdataset, dataset))
    x_dim = get_data_dim(dataset)
    path = data_path_dict[dataset]

    prefix = subdataset
    train_files = glob(os.path.join(path, prefix + "_train.pkl"))
    test_files = glob(os.path.join(path, prefix + "_test.pkl"))
    label_files = glob(os.path.join(path, prefix + "_test_label.pkl"))

    print("{} files found.".format(len(train_files)))

    data_dict = defaultdict(dict)
    data_dict["dim"] = x_dim if use_dim == "all" else 1

    train_data_list = []
    for idx, f_name in enumerate(train_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        train_data = pickle.load(f).reshape((-1, x_dim))
        f.close()
        if use_dim != "all":
            train_data = train_data[:nrows, use_dim].reshape(-1, 1)
        if len(train_data) > 0:
            train_data_list.append(train_data[:nrows])
    data_dict["train"] = np.concatenate(train_data_list, axis=0)

    test_data_list = []
    for idx, f_name in enumerate(test_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))
        f.close()
        if use_dim != "all":
            test_data = test_data[:nrows, use_dim].reshape(-1, 1)
        if len(test_data) > 0:
            test_data_list.append(test_data[:nrows])
    data_dict["test"] = np.concatenate(test_data_list, axis=0)

    label_data_list = []
    for idx, f_name in enumerate(label_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        label_data = pickle.load(f)[:nrows]
        f.close()
        if len(label_data) > 0:
            label_data_list.append(label_data)
    data_dict["test_labels"] = np.concatenate(label_data_list, axis=0)

    train_data = preprocess(data_dict["train"])
    test_data = preprocess(data_dict["test"])
    # return data_dict
    return (train_data, None), (test_data, data_dict["test_labels"])


def save_z(z, filename="z"):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + "_" + str(i) + ".txt", "w") as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write("%f " % (z[j][i][k]))
                file.write("\n")
    i = z.shape[1] - 1
    with open(filename + "_" + str(i) + ".txt", "w") as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write("%f " % (z[j][i][k]))
            file.write("\n")


# def get_data_dim(dataset):
#     if dataset == "SMAP":
#         return 25
#     elif dataset == "MSL":
#         return 55
#     elif str(dataset).startswith("machine"):
#         return 38
#     else:
#         raise ValueError("unknown dataset " + str(dataset))


def get_data(
    dataset,
    max_train_size=None,
    max_test_size=None,
    print_log=True,
    do_preprocess=True,
    train_start=0,
    test_start=0,
):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """returns normalized and standardized data."""

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be a 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains null values. Will be replaced with 0")
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print("Data normalized")

    return df


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
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)


def iter_thresholds(score, label):
    best_f1 = -float("inf")
    best_theta = None
    best_adjust = None
    best_raw = None
    for anomaly_ratio in np.linspace(1e-3, 1, 500)[0:1]:
        info_save = {}
        adjusted_anomaly, raw_predict, threshold = score2pred(
            score, label, percent=100 * (1 - anomaly_ratio), adjust=False
        )

        f1 = f1_score(adjusted_anomaly, label)
        if f1 > best_f1:
            best_f1 = f1
            best_adjust = adjusted_anomaly
            best_raw = raw_predict
            best_theta = threshold
    return best_f1, best_theta, best_adjust, best_raw


def score2pred(
    score,
    label,
    percent=None,
    threshold=None,
    pred=None,
    calc_latency=False,
    adjust=True,
):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is higher than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if score is not None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        if percent is not None:
            threshold = np.percentile(score, percent)
            # print("Threshold for {} percent is: {:.2f}".format(percent, threshold))
            predict = score > threshold
        elif threshold is not None:
            predict = score > threshold
    else:
        predict = pred

    if not adjust:
        return predict, predict, threshold

    raw_predict = copy.deepcopy(predict)

    actual = label == 1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict, raw_predict, threshold


subdatasets = {
    "SMD": ["machine-1-{}".format(i) for i in range(1, 9)]
    + ["machine-2-{}".format(i) for i in range(1, 10)]
    + ["machine-3-{}".format(i) for i in range(1, 12)],
    "SMAP": [
        "P-1",
        "S-1",
        "E-1",
        "E-2",
        "E-3",
        "E-4",
        "E-5",
        "E-6",
        "E-7",
        "E-8",
        "E-9",
        "E-10",
        "E-11",
        "E-12",
        "E-13",
        "A-1",
        "D-1",
        "P-2",
        "P-3",
        "D-2",
        "D-3",
        "D-4",
        "A-2",
        "A-3",
        "A-4",
        "G-1",
        "G-2",
        "D-5",
        "D-6",
        "D-7",
        "F-1",
        "P-4",
        "G-3",
        "T-1",
        "T-2",
        "D-8",
        "D-9",
        "F-2",
        "G-4",
        "T-3",
        "D-11",
        "D-12",
        "B-1",
        "G-6",
        "G-7",
        "P-7",
        "R-1",
        "A-5",
        "A-6",
        "A-7",
        "D-13",
        "P-2",
        "A-8",
        "A-9",
        "F-3",
    ],
    "MSL": [
        "M-6",
        "M-1",
        "M-2",
        "S-2",
        "P-10",
        "T-4",
        "T-5",
        "F-7",
        "M-3",
        "M-4",
        "M-5",
        "P-15",
        "C-1",
        "C-2",
        "T-12",
        "T-13",
        "F-4",
        "F-5",
        "D-14",
        "T-9",
        "P-14",
        "T-8",
        "P-11",
        "D-15",
        "D-16",
        "M-7",
        "F-8",
    ],
}