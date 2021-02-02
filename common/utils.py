import numpy as np
import json
import logging
import h5py
import nni
import random
import os
import torch


def seed_everything(seed=1029):
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
            # logging.info("Threshold for {} percent is: {:.2f}".format(percent, threshold))
            predict = score > threshold
        elif threshold is not None:
            predict = score > threshold
    else:
        predict = pred

    if not adjust:
        return predict
    else:
        actual = label > 0.1
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
            return predict