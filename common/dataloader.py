import logging
import os
import pickle
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
from IPython import embed

data_path_dict = {
    "SMD": "./datasets/anomaly/SMD/processed",
    "SMD12": "./datasets/anomaly/SMD12",
    "OMI": "./datasets/anomaly/OMI",
    "SMAP": "./datasets/anomaly/SMAP-MSL/processed_SMAP",
    "MSL": "./datasets/anomaly/SMAP-MSL/processed_MSL",
    "WADI": "./datasets/anomaly/WADI/processed",
    "SWAT": "./datasets/anomaly/SWAT/processed",
    "WADI_SPLIT": "./datasets/anomaly/WADI_SPLIT/processed",
    "SWAT_SPLIT": "./datasets/anomaly/SWAT_SPLIT/processed",
    "HUAWEI": "./datasets/anomaly/HW"
}


def get_data_dim(dataset, subdataset=""):
    if "SMAP" in dataset:
        return 25
    elif "MSL" in dataset:
        return 55
    elif "SMD" in dataset:
        return 38
    elif "WADI" in dataset:
        return 93
    elif "SWAT" in dataset:
        return 40
    elif "HUAWEI" in dataset:
        return {
            "aa":30, # change to hashid
            "bb":30,
            "cc":30,
        }[subdataset]
    else:
        raise ValueError("unknown dataset " + str(dataset))


def load_dataset(dataset, subdataset, use_dim="all", root_dir="../", nrows=None):
    """
    use_dim: dimension used in multivariate timeseries
    """
    print("Loading {} of {} dataset".format(subdataset, dataset))
    path = data_path_dict[dataset]
    prefix = subdataset
    train_files = glob(os.path.join(root_dir, path, prefix + "_train.pkl"))
    test_files = glob(os.path.join(root_dir, path, prefix + "_test.pkl"))
    label_files = glob(os.path.join(root_dir, path, prefix + "_test_label.pkl"))

    print(os.path.join(root_dir, path, prefix + "_train.pkl"))
    logging.info("{} files found.".format(len(train_files)))

    data_dict = defaultdict(dict)

    train_data_list = []
    for idx, f_name in enumerate(train_files):
        f = open(f_name, "rb")
        train_data = pickle.load(f)
        f.close()
        if use_dim != "all":
            train_data = train_data[:, use_dim].reshape(-1, 1)
        if len(train_data) > 0:
            train_data_list.append(train_data)
    data_dict["train"] = np.concatenate(train_data_list, axis=0)[:nrows]

    test_data_list = []
    for idx, f_name in enumerate(test_files):
        f = open(f_name, "rb")
        test_data = pickle.load(f)
        f.close()
        if use_dim != "all":
            test_data = test_data[:, use_dim].reshape(-1, 1)
        if len(test_data) > 0:
            test_data_list.append(test_data)
    data_dict["test"] = np.concatenate(test_data_list, axis=0)[:nrows]

    label_data_list = []
    for idx, f_name in enumerate(label_files):
        f = open(f_name, "rb")
        label_data = pickle.load(f)
        f.close()
        if len(label_data) > 0:
            label_data_list.append(label_data)
    data_dict["test_labels"] = np.concatenate(label_data_list, axis=0)[:nrows]

    for k, v in data_dict.items():
        if k == "dim":
            continue
        print("Shape of {} is {}.".format(k, v.shape))
    return data_dict
