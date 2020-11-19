import logging
import os
import pickle
import torch
import numpy as np
from collections import defaultdict
from common.sliding import BatchSlidingWindow
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler, RobustScaler
from IPython import embed

class preprocessor():
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


    def discretize(self, data_dict, n_bins=1000, mode="train"):
        discretized_dict = defaultdict(dict)
        for data_name, sub_dict in data_dict.items():
            if not isinstance(sub_dict, dict):
                discretized_dict[data_name] = sub_dict
                continue
            
            if isinstance(n_bins, dict):
                n_bins_ = n_bins[data_name]
            else:
                n_bins_ = n_bins

            if mode=="test":
                est = self.discretizer_list[data_name]
                if "train" in sub_dict:
                    train = sub_dict["train"]
                    train_ = est.transform(train)
                    discretized_dict[data_name]["train"] = train_.astype(int)
            elif mode=="train":
                # fit_transform using train
                train = sub_dict["train"] # shape: time x dim
                est = KBinsDiscretizer(n_bins=n_bins_, encode='ordinal', strategy='uniform')
                train_ = est.fit_transform(train)
                discretized_dict[data_name]["train"] = train_.astype(int)
                self.discretizer_list[data_name] = est

            # transform test
            test = sub_dict["test"]
            test_ = est.transform(test)

            # assign back
            discretized_dict[data_name]["test"] = test_.astype(int)
        return discretized_dict

    def build_vocab(self, data_dict):
        max_index = -float("inf")
        for data_name, sub_dict in data_dict.items():
            if not isinstance(sub_dict, dict):
                continue

            index = np.max(sub_dict["train"].reshape(-1))
            if index > max_index:
                max_index = index

        self.vocab_size = max_index+1
        logging.info("# of Discretized tokens: {}".format(self.vocab_size))

        return self.vocab_size


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


def generate_windows(data_dict, window_size=100, nrows=None):
    train_windows = []
    test_windows = []
    results = {}

    for data_name, sub_dict in data_dict.items():
        if not isinstance(sub_dict, dict): continue
        logging.info("Generating sliding windows for dataset [{}]".format(data_name))
        test = sub_dict["test"][0:nrows]
        test_label = None if "test_label" not in sub_dict else sub_dict["test_label"][0:nrows]
        test_win = BatchSlidingWindow(test.shape[0], window_size=window_size, batch_size=1000, shuffle=False).get_windows(test, test_label)
        test_windows.append(test_win)

        if "train" in sub_dict:
            train = sub_dict["train"][0:nrows]
            train_win = BatchSlidingWindow(train.shape[0], window_size=window_size, batch_size=1000, shuffle=False).get_windows(train)
            train_windows.append(train_win)
            

    if train_windows:
        train_windows = torch.cat(train_windows,dim=0)
        results["train_windows"] = train_windows.cpu().numpy()
    if test_windows:
        test_windows = torch.cat(test_windows,dim=0)
        if test_label is not None:
            test_windows, test_labels = test_windows[:, 0:-1], test_windows[:, -1]
            results["test_windows"] = test_windows.cpu().numpy()
            results["test_labels"] = test_labels.cpu().numpy()
        else:
            results["test_windows"] = test_windows.cpu().numpy()
    return results
        
        
