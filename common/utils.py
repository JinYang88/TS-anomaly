import numpy as np
import json
import logging
import h5py
import nni
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def seed_everything(seed=1029):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_hdf5(infile):
    logging.info("Loading hdf5 from {}".format(infile))
    with h5py.File(infile, "r") as f:
        return {key: f[key][:] for key in list(f.keys())}


def save_hdf5(outfile, arr_dict):
    logging.info("Saving hdf5 to {}".format(outfile))
    with h5py.File(outfile, "w") as f:
        for key in arr_dict.keys():
            f.create_dataset(key, data=arr_dict[key])


def print_to_json(data):
    new_data = dict((k, str(v)) for k, v in data.items())
    return json.dumps(new_data, indent=4, sort_keys=True)


def update_from_nni_params(params, nni_params):
    if nni_params:
        params.update(nni_params)
    return params


def iter_thresholds(score, label):
    best_f1 = -float("inf")
    best_theta = None
    best_adjust = None
    best_raw = None
    for anomaly_ratio in np.linspace(1e-3, 1, 500):
        info_save = {}
        adjusted_anomaly, raw_predict, threshold = score2pred(
            score, label, percent=100 * (1 - anomaly_ratio)
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
            # logging.info("Threshold for {} percent is: {:.2f}".format(percent, threshold))
            predict = score > threshold
        elif threshold is not None:
            predict = score > threshold
    else:
        predict = pred

    import copy

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