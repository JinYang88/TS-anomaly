import logging
from collections import defaultdict

import torch
import numpy as np
from common.sliding import BatchSlidingWindow
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler, RobustScaler

def normalize(data_dict, method="minmax"):
    # method: minmax, standard, robust
    normalized_dict = defaultdict(dict)
    for data_name, sub_dict in data_dict.items():
        if not isinstance(sub_dict, dict):
            normalized_dict[data_name] = sub_dict
            continue
        
        if isinstance(n_bins, dict):
            n_bins_ = n_bins[data_name]
        else:
            n_bins_ = n_bins

        # fit_transform using train
        train = sub_dict["train"] # shape: time x dim
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
        normalized_dict[data_name]["train"] = train_.astype(int)
        normalized_dict[data_name]["test"] = test_.astype(int)
    return normalized_dict

def discretize(data_dict, n_bins=1000):
    discretized_dict = defaultdict(dict)
    for data_name, sub_dict in data_dict.items():
        if not isinstance(sub_dict, dict):
            discretized_dict[data_name] = sub_dict
            continue
        
        if isinstance(n_bins, dict):
            n_bins_ = n_bins[data_name]
        else:
            n_bins_ = n_bins

        # fit_transform using train
        train = sub_dict["train"] # shape: time x dim
        est = KBinsDiscretizer(n_bins=n_bins_, encode='ordinal', strategy='uniform')
        train_ = est.fit_transform(train)

        # transform test
        test = sub_dict["test"]
        test_ = est.transform(test)

        # assign back
        discretized_dict[data_name]["train"] = train_.astype(int)
        discretized_dict[data_name]["test"] = test_.astype(int)
    return discretized_dict

def build_vocab(data_dict):
    max_index = -float("inf")
    for data_name, sub_dict in data_dict.items():
        if not isinstance(sub_dict, dict):
            continue

        index = np.max(sub_dict["train"].reshape(-1))
        if index > max_index:
            max_index = index

    logging.info("# of Discretized tokens: {}".format(max_index+1))
    return max_index+1

def preprocess_SMD(data_dict, window_size=100):
    train_windows = []
    test_windows = []
    
    for data_name, sub_dict in data_dict.items():
        if not isinstance(sub_dict, dict): continue
        train = sub_dict["train"]
        test = sub_dict["test"]
        train_win = BatchSlidingWindow(train.shape[0], window_size=window_size, batch_size=1000, shuffle=False).get_windows(train)
        test_win = BatchSlidingWindow(test.shape[0], window_size=window_size, batch_size=1000, shuffle=False).get_windows(test)

        train_windows.append(train_win)
        test_windows.append(test_win)
    
    return torch.cat(train_windows,dim=0), torch.cat(test_windows,dim=0)
        
        
