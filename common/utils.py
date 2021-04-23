import numpy as np
import json
import logging
import h5py
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def set_device(gpu=-1):
    import torch

    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def pprint(d, indent=0):
    d = sorted([(k, v) for k, v in d.items()], key=lambda x: x[0])
    for key, value in d:
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pprint(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(round(value, 4)))


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
