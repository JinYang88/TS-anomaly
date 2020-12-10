#!/usr/bin/env python
# coding: utf-8

import json
import logging
import math
import os
import sys

sys.path.append("../")
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import pandas as pd
import torch
from common import data_preprocess
from common.dataloader import (
    load_CSV_dataset,
    load_kddcup_dataset,
    load_SMAP_MSL_dataset,
    load_SMD_dataset,
)
from common.utils import print_to_json
from IPython import embed
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def adjust_predicts(
    score,
    label,
    percent=None,
    threshold=None,
    pred=None,
    calc_latency=False,
    verbose=False,
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
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        if percent is not None:
            threshold = np.percentile(score, percent)
            if verbose:
                print("Threshold for {} percent is: {:.2f}".format(percent, threshold))
            predict = score > threshold
            if verbose:
                print("{:.3f}% is anomaly".format(100 * predict.sum() / len(predict)))
        elif threshold is not None:
            predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
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
        return predict


def get_windows(ts, labels=None, dim=None, window_size=128, stride=None):
    if stride is None:
        stride = window_size
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
        return np.array(windows), np.array(label_windows)
    else:
        return np.array(windows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Anomaly detection with traditional ways"
    )
    parser.add_argument("--dataset", type=str, default="SMD_1-1")
    parser.add_argument("--inter", action="store_true")
    parser.add_argument("--norm", action="store_true")

    args = parser.parse_args()
    # datasets/anomaly/SMD/processed/machine-1-1_test.pkl
    nrows = None
    dataset = args.dataset

    if dataset.lower().startswith("smd"):
        data_dict = load_SMD_dataset(
            "../datasets/anomaly/SMD/processed",
            "machine-{}".format(dataset.split("_")[-1]),
        )
    elif dataset.lower() == "kddcup":
        data_dict = load_kddcup_dataset(path="../datasets/anomaly/Kddcup9")
    elif dataset.lower() == "smap" or dataset.lower() == "msl":
        data_dict = load_SMAP_MSL_dataset("../datasets/anomaly/SMAP-MSL", dataset)

    if nrows is not None:
        data_dict["test_label"] = data_dict["test_label"][0:nrows]
        data_dict["train"] = data_dict["train"][0:nrows]
        data_dict["test"] = data_dict["test"][0:nrows]

    print(
        "Anomaly ratio for {} = {:.3f}".format(
            dataset, data_dict["test_label"].sum() / len(data_dict["test_label"])
        )
    )
    print("Length for train: {} = {}".format(dataset, len(data_dict["train"])))
    print("Length for test: {} = {}".format(dataset, len(data_dict["test"])))

    if args.inter:
        print("Doing feature interaction")
        pf = PolynomialFeatures(include_bias=True, degree=2)
        data_dict["train"] = pf.fit_transform(data_dict["train"])
        data_dict["test"] = pf.fit_transform(data_dict["test"])

    if args.norm:
        print("Doing feature normalization")
        sd = StandardScaler()
        data_dict["train"] = sd.fit_transform(data_dict["train"])
        data_dict["test"] = sd.transform(data_dict["train"])

    # Get windows
    # window_size = 100
    # train_df = pd.DataFrame(data_dict["train"])
    # test_df = pd.DataFrame(data_dict["test"])
    # test_label = pd.DataFrame(data_dict["test_label"])
    # train_windows = get_windows(data_dict["train"], dim=None, window_size=window_size)
    # test_windows, test_labels = get_windows(data_dict["test"], labels=data_dict["test_label"], dim=None, window_size=window_size)

    info_save_list = []
    for linkage in ["complete", "single", "average"]:
        n_clusters = 30
        clusters = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(
            data_dict["train"]
        )
        print("Clustering finished.")

        cluster_ts = defaultdict(list)
        for k, v in zip(clusters.labels_, data_dict["train"]):
            cluster_ts[k].append(v)
        centers = np.array([np.array(v).mean(axis=0) for k, v in cluster_ts.items()])
        print("Computing centers done")

        for anomaly_ratio in np.linspace(1e-3, 0.5, 50):
            info_save = {}
            info_save["linkage"] = linkage
            info_save["inter"] = args.inter
            info_save["norm"] = args.norm
            test_dist_mat = cosine_similarity(data_dict["test"], centers)
            anomaly_rate = 1 - test_dist_mat.max(axis=1)
            adjusted_anomaly = adjust_predicts(
                anomaly_rate, data_dict["test_label"], percent=100 * (1 - anomaly_ratio)
            )
            f1 = f1_score(adjusted_anomaly, data_dict["test_label"])
            rc = recall_score(adjusted_anomaly, data_dict["test_label"])
            pr = precision_score(adjusted_anomaly, data_dict["test_label"])
            info_save["F1"] = f1
            info_save["Recall"] = rc
            info_save["Precision"] = pr
            info_save["Anomaly_ratio"] = anomaly_ratio
            #     print("F1: {:.2f} RC: {:.2f} PR: {:.2f}".format(f1, rc, pr))
            info_save_list.append(info_save)

    df = pd.DataFrame(info_save_list)
    filename = "exp_results_" + dataset
    if args.inter:
        filename += "_inter"
    if args.norm:
        filename += "_norm"
    df.to_csv(filename + ".csv", index=False)
