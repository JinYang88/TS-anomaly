import pickle
import os
import pandas as pd
import numpy as np
import weka.core.jvm
import weka.core.converters

from IPython import embed
from glob import glob
from collections import defaultdict

def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif dataset == 'SMD' or str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset '+str(dataset))

def load_SMAP_MSL_dataset(path, dataset="MSL", use_dim="all"):
    data_dict = defaultdict(dict)
    if use_dim == "all":
        data_dict["dim"] = get_data_dim(dataset)
    else:
        data_dict["dim"] = 1
    pkl_files = glob(os.path.join(path, "pkls_" + dataset, "*.pkl"))

    for f in pkl_files:
        basename = os.path.basename(f)
        with open(f, "rb") as fr:
            array = pickle.load(fr)
        if basename.startswith(dataset+"_train"):
            data_dict[dataset]["train"] = array if use_dim == "all" else array[:, use_dim]
        if basename.startswith(dataset+"_test_label"):
            data_dict[dataset]["test_label"] = array
        if basename.startswith(dataset+"_test"):
            data_dict[dataset]["test"] = array if use_dim == "all" else array[:, use_dim]

    return data_dict

def load_CSV_dataset(path, dataset="all", test_ratio=0.2):
    df = pd.read_csv(path)
    train_size = int(df.shape[0] * (1-test_ratio))
    train_df = df[:train_size]
    test_df = df[train_size:]

    data_dict = defaultdict(dict)
    data_dict["dim"] = 1
    columns = train_df.columns if dataset == "all" else [dataset]
    for f_name in columns:
        data_dict[f_name]["train"] = np.array(train_df[f_name]).reshape(-1,1)
        data_dict[f_name]["test"] = np.array(test_df[f_name]).reshape(-1,1)
    return data_dict

def load_SMD_dataset(path, dataset, use_dim="all"):
    x_dim = get_data_dim(dataset)

    if str(dataset).startswith('machine'):
        prefix = dataset
    else:
        prefix = "*"

    train_files = glob(os.path.join(path, prefix + "_train.pkl"))
    test_files = glob(os.path.join(path, prefix + "_test.pkl"))
    label_files = glob(os.path.join(path, prefix + "_test_label.pkl"))

    data_dict = defaultdict(dict)
    data_dict["dim"] = x_dim if use_dim == "all" else 1
    for idx, f_name in enumerate(train_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        train_data = pickle.load(f).reshape((-1, x_dim))
        f.close()
        if use_dim != "all":
            train_data = train_data[:, use_dim].reshape(-1, 1)
        data_dict[machine_name]["train"] = train_data

    for idx, f_name in enumerate(test_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))
        f.close()
        if use_dim != "all":
            test_data = test_data[:, use_dim].reshape(-1, 1)
        data_dict[machine_name]["test"] = test_data

    for idx, f_name in enumerate(label_files):
        machine_name = os.path.basename(f_name).split("_")[0]
        f = open(f_name, "rb")
        test_label = pickle.load(f).reshape((-1))
        f.close()
        data_dict[machine_name]["test_label"] = test_label
    
    return data_dict

    
def load_UCR_dataset(path, dataset):
    """
    Loads the UCR dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = np.expand_dims(train_array[:, 1:], 1).astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = np.expand_dims(test_array[:, 1:], 1).astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train, train_labels, test, test_labels
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    mean = np.nanmean(np.concatenate([train, test]))
    var = np.nanvar(np.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels



