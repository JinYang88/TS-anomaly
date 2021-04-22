import sys
import os

sys.path.append("../..")
import pickle
from glob import glob
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from common.evaluation import iter_thresholds, point_adjustment
from dtaidistance import dtw
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from common.spot import SPOT
from common.evaluation import iter_thresholds
import traceback
import argparse

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

root_dir = "../../benchmark/benchmarking_results"
subdataset_num = {
    "SMD": 28,
    "SMAP": 54,
    "MSL": 27,
    "SWAT": 1,
    "WADI": 1,
    "SWAT_SPLIT": 3,
    "WADI_SPLIT": 3,
}

metric_key_num = 10


def get_dir_list(path):
    dir_list = []
    for item in os.listdir(path):
        directory = os.path.join(path, item).replace("\\", "/")
        if os.path.isdir(directory):
            dir_list.append(item)
    return dir_list


data = []
res_dict = defaultdict(dict)
est = MinMaxScaler(clip=True)
# for model in ["lstm"]:
for model in get_dir_list(root_dir):
    print(model)
    model_dirs = os.path.join(root_dir, model)  # do not use root_dir + model
    for hash_id in get_dir_list(model_dirs):
        try:
            hash_id_dirs = model_dirs + "/" + hash_id
            for dataset in get_dir_list(hash_id_dirs):
                dataset_dirs = hash_id_dirs + "/" + dataset
                subdataset_count = 0
                miss_metrics = False
                miss_time = False
                miss_key = False
                avg_adj_f1 = 0
                avg_raw_f1 = 0
                train_time = 0
                test_time = 0
                for subdataset in get_dir_list(dataset_dirs):
                    subdataset_dir = dataset_dirs + "/" + subdataset
                    metrics_dir = subdataset_dir + "/" + "metrics.json"
                    time_dir = subdataset_dir + "/" + "time.json"
                    if not os.path.exists(time_dir):
                        miss_time = True
                    else:
                        if not os.path.exists(metrics_dir):
                            miss_metrics = True
                        else:
                            with open(metrics_dir) as fr:
                                metrics_dict = json.load(fr)
                            with open(time_dir) as fr:
                                time_dict = json.load(fr)
                            if len(metrics_dict) != metric_key_num:
                                miss_key = True
                            else:
                                avg_adj_f1 += metrics_dict["adj_f1"]
                                avg_raw_f1 += metrics_dict["raw_f1"]
                                train_time += time_dict["train"]
                                test_time += time_dict["test"]
                                #                             print(subdataset, score_dict["train"])
                                score_dict = np.load(
                                    os.path.join(subdataset_dir, "anomaly_score.npz"),
                                    allow_pickle=True,
                                )["arr_0"].item()
                                anomaly_score_train = est.fit_transform(
                                    score_dict["train"].reshape(-1, 1)
                                )
                                anomaly_score_test = est.fit_transform(
                                    score_dict["test"].reshape(-1, 1)
                                )

                                label = np.load(
                                    os.path.join(subdataset_dir, "anomaly_label.npz")
                                )["arr_0"].astype(int)
                                res_dict[hash_id][subdataset] = {
                                    "train_score": anomaly_score_train,
                                    "test_score": anomaly_score_test,
                                    "label": label,
                                }
                                subdataset_count += 1
                with open(os.path.join(hash_id_dirs, "params.json")) as fr:
                    cmd = json.load(fr)["cmd"]
                avg_adj_f1 = avg_adj_f1 / subdataset_num[dataset]
                avg_raw_f1 = avg_raw_f1 / subdataset_num[dataset]

                data.append(
                    [
                        model,
                        dataset,
                        subdataset_count,
                        hash_id,
                        avg_adj_f1,
                        avg_raw_f1,
                        train_time,
                        test_time,
                        cmd,
                    ]
                )
        except:
            print(f"failed on {hash_id}")
print("Done.")

data_df = pd.DataFrame(
    data,
    columns=[
        "model",
        "dataset",
        "count",
        "hash_id",
        "adj_f1",
        "raw_f1",
        "train_time",
        "train_time",
        "cmd",
    ],
)
data_df["dataset"] = data_df["dataset"].map(
    lambda x: "WADI" if x == "WADI_SPLIT" else x
)
best_data = data_df.loc[data_df.groupby(["model", "dataset"])["adj_f1"].idxmax()]


q = 1e-3


def iter_pot_for_hashid(hash_id, res_dict):
    #     print(f"Processing {hash_id}")
    pot_metrics = []
    iter_metrics = []
    for subdataset, value_dict in list(res_dict[hash_id].items()):
        train_score = value_dict["train_score"]
        test_score = value_dict["test_score"]
        #         print(train_score.shape, test_score.shape)
        label = value_dict["label"]
        best_metric_iter, best_theta, best_adjust, best_raw = iter_thresholds(
            test_score, label, metric="f1", adjustment=True
        )
        iter_metrics.append(best_metric_iter)
        try:
            s = SPOT(q=q)  # SPOT object
            s.fit(train_score, test_score)  # data import
            for level in [0.001, 0.003, 0.005, 0.01, 0.1, 0.07, 0.0001, 0.00001]:
                try:
                    s.initialize(level=level, min_extrema=False)  # initialization step
                except:
                    pass
            ret = s.run(dynamic=False)  # run
            pot_th = np.mean(ret["thresholds"])
            (
                best_metric_pot,
                best_theta_pot,
                best_adjust_pot,
                best_raw_pot,
            ) = iter_thresholds(
                test_score, label, metric="f1", adjustment=True, threshold=pot_th
            )
            pot_metrics.append(best_metric_pot)
        except:
            print(traceback.format_exc())
            pot_metrics.append(iter_metrics[-1])
            print(f"failed on {hash_id}")
    pot_mean = np.mean(pot_metrics)
    iter_mean = np.mean(iter_metrics)
    print(pot_mean, iter_mean)
    return pot_mean, iter_mean


### test case ###
# x = best_data["hash_id"][0]
# x = "88e6097e"
# iter_pot_for_hashid(x, res_dict)
##################

pot_iter_results = best_data["hash_id"].map(lambda x: iter_pot_for_hashid(x, res_dict))
pf1, af1 = zip(*pot_iter_results)
best_data["pf1"] = pf1
best_data["af1"] = af1

best_data.to_csv(f"all_best_data_df_{q}.csv", index=False)

with open("th_save_dict.pkl", "wb") as fw:
    pickle.dump(th_save_dict, fw)