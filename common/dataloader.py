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


def load_dataset(dataset, subdataset, use_dim="all", root_dir="../"):
    """
    use_dim: dimension used in multivariate timeseries
    """
    logging.info("Loading {} of {} dataset".format(subdataset, dataset))
    x_dim = get_data_dim(dataset)
    path = data_path_dict[dataset]

    prefix = subdataset
    train_files = glob(os.path.join(root_dir, path, prefix + "_train.pkl"))
    test_files = glob(os.path.join(root_dir, path, prefix + "_test.pkl"))
    label_files = glob(os.path.join(root_dir, path, prefix + "_test_label.pkl"))

    logging.info("{} files found.".format(len(train_files)))

    data_dict = defaultdict(dict)
    data_dict["dim"] = x_dim if use_dim == "all" else 1

    train_data_list = []
    for idx, f_name in enumerate(train_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        train_data = pickle.load(f).reshape((-1, x_dim))
        f.close()
        if use_dim != "all":
            train_data = train_data[:, use_dim].reshape(-1, 1)
        if len(train_data) > 0:
            train_data_list.append(train_data)
    data_dict["train"] = np.concatenate(train_data_list, axis=0)

    test_data_list = []
    for idx, f_name in enumerate(test_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))
        f.close()
        if use_dim != "all":
            test_data = test_data[:, use_dim].reshape(-1, 1)
        if len(test_data) > 0:
            test_data_list.append(test_data)
    data_dict["test"] = np.concatenate(test_data_list, axis=0)

    label_data_list = []
    for idx, f_name in enumerate(label_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        label_data = pickle.load(f)
        f.close()
        if len(label_data) > 0:
            label_data_list.append(label_data)
    data_dict["test_labels"] = np.concatenate(label_data_list, axis=0)

    return data_dict
