# %load_ext autoreload
# %autoreload 2
import sys
import os
sys.path.append("../..")

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

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

root_dir = '../../benchmark/benchmarking_results'
subdataset_num = {'SMD': 28, 'SMAP': 54, 'MSL': 27, 'SWAT': 1, 'WADI': 1,
                  'SWAT_SPLIT': 3, 'WADI_SPLIT': 3}

metric_key_num = 10



def get_dir_list(path):
    dir_list = []
    for item in os.listdir(path):
        directory = os.path.join(path, item).replace('\\', '/')
        if os.path.isdir(directory):
            dir_list.append(item)
    return dir_list

def normalize_score(score):
    return est.fit_transform(np.nan_to_num(score).reshape(-1,1)) 
    
data = []
res_dict = defaultdict(dict)
est = MinMaxScaler(clip=True)
# for model in ["PCA"]:
for model in get_dir_list(root_dir):
    model_dirs = os.path.join(root_dir, model) # do not use root_dir + model
    for hash_id in get_dir_list(model_dirs):
        hash_id_dirs = model_dirs + '/' + hash_id
        for dataset in get_dir_list(hash_id_dirs):
            dataset_dirs = hash_id_dirs + '/' + dataset
            subdataset_count = 0
            miss_metrics = False
            miss_time = False
            miss_key = False
            avg_adj_f1 = 0
            avg_raw_f1 = 0
            train_time = 0
            test_time = 0
            for subdataset in get_dir_list(dataset_dirs):
                if subdataset=="T-1":continue
                subdataset_dir = dataset_dirs + '/' + subdataset
                metrics_dir = subdataset_dir + '/' + 'metrics.json'
                time_dir = subdataset_dir + '/' + 'time.json'
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
                            try:
                                
                                avg_adj_f1 += metrics_dict['adj_f1']
                                avg_raw_f1 += metrics_dict['raw_f1']
                                train_time += time_dict['train']
                                test_time += time_dict['test']
                                score_dict = np.load(os.path.join(subdataset_dir, "anomaly_score.npz"), allow_pickle=True)["arr_0"].item()
                                np.nan_to_num(score_dict["train"])
                                anomaly_score_train = normalize_score(score_dict["train"])
                                anomaly_score_test = normalize_score(score_dict["test"])

                                label = np.load(os.path.join(subdataset_dir, "anomaly_label.npz"))["arr_0"].astype(int)
                                res_dict[(hash_id, model)][subdataset] = {"train_score": anomaly_score_train, "test_score": anomaly_score_test, "label": label}
                                subdataset_count += 1
                            except:
#                                 print(subdataset, score_dict["train"].max())
                                print(traceback.format_exc())
                                print(model, dataset, subdataset, "failed")
            with open(os.path.join(hash_id_dirs, "params.json")) as fr:
                cmd = json.load(fr)["cmd"]
            avg_adj_f1 = avg_adj_f1 / subdataset_num[dataset]
            avg_raw_f1 = avg_raw_f1 / subdataset_num[dataset]

#             if subdataset_count >= 0.7*subdataset_num[dataset] and (not miss_time) and (not miss_metrics)\
#                     and (not miss_key):
            data.append([model, dataset, subdataset_count, hash_id, avg_adj_f1, avg_raw_f1, train_time, test_time, cmd])

print("Done.")


data_df = pd.DataFrame(data, columns=['model', 'dataset', 'count' , 'hash_id', 'adj_f1', 'raw_f1', "train_time", 'test_time', "cmd"])
data_df["dataset"] = data_df["dataset"].map(lambda x: "WADI" if x=="WADI_SPLIT" else x)
best_data = data_df.loc[data_df.groupby(["model", "dataset"])["adj_f1"].idxmax()]

from sklearn.cluster import AgglomerativeClustering
def compute_support(score, label, dtype="normal"):
    if dtype == "normal":
        score_idx = np.arange(len(score))[(label==0).astype(bool)]
    elif dtype == "anomaly":
        score_idx = np.arange(len(score))[(label==1).astype(bool)]

    clusters = []
    dscore = score[score_idx]
    print("Clustering...")
    clustering = AgglomerativeClustering(affinity="l1", linkage="complete").fit(dscore)
    cluster_labels = clustering.labels_

    for label in range(len(set(cluster_labels))):
        clusters.append(dscore[cluster_labels == label])
    max_label = max(enumerate(clusters), key=lambda x: np.mean(x[1]))[0]
    
    max_cluster = clusters[max_label] 
    std = np.std(max_cluster)
    mean = np.mean(max_cluster)
    original_idx = score_idx[cluster_labels==max_label]

    
    ## plot internal
#     plot_x = np.arange(len(dscore))
#     scatter_x = plot_x[cluster_labels==max_label]
#     plt.figure()
#     plt.plot(plot_x, dscore)
#     plt.scatter(scatter_x, max_cluster, c="r")
#     plt.hlines(mean, 0, cluster_labels.shape[0], "r", label=f"mean:{mean:.3f}")
#     mean_arr = np.array([mean] * len(plot_x))
#     plt.fill_between(plot_x, mean_arr-std, mean_arr+std, alpha=0.2, facecolor = "green")
#     plt.show()
    
    return_dict = {
        "mean": mean,
        "std": std,
        "idx": original_idx
    }
    return return_dict


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def compute_salience(score, label, plot=False, ax=None, fig_saving_path=""):
    total_indice = np.arange(len(score))
    score_n = score[~label.astype(bool)]
    score_a = score[label.astype(bool)]

    score_n_idx = total_indice[~label.astype(bool)]
    n_dict = compute_support(score, label, "normal")
    salient_score_n = score[n_dict["idx"]]

    score_a_idx = total_indice[label.astype(bool)]
    a_dict = compute_support(score, label, "anomaly")
    salient_score_a = score[a_dict["idx"]]
    
    a_upper = a_dict["mean"] + a_dict["std"]
    a_lower = a_dict["mean"] - a_dict["std"]
    n_upper = n_dict["mean"] + n_dict["std"]
    n_lower = n_dict["mean"] - n_dict["std"]
    
#     overlapping = (n_upper - a_lower if n_upper >= a_lower else 0) / 2*(min(a_dict["std"], n_dict["std"]))
    overlapping = (n_upper - a_lower if n_upper >= a_lower else 0) / (max(a_upper, n_upper) - min(a_lower, n_lower))
    non_overlapping = 1 - overlapping
    
#     soft_a_count  = sigmoid(len(a_dict["idx"]))
#     soft_n_count  = sigmoid(len(n_dict["idx"]))
#     a_count_ratio = soft_a_count / (soft_n_count + soft_a_count)

    a_count_ratio = sigmoid(len(a_dict["idx"]) / (len(a_dict["idx"]) + len(n_dict["idx"])))
    n_count_ratio = sigmoid(len(n_dict["idx"]) / (len(a_dict["idx"]) + len(n_dict["idx"])))

    a_count_ratio = a_count_ratio / (a_count_ratio + n_count_ratio)
    n_count_ratio = n_count_ratio / (a_count_ratio + n_count_ratio)
    
    salience = non_overlapping * (a_count_ratio * a_dict["mean"] - n_count_ratio * n_dict["mean"])
    
#     print(f"a count: {len(a_dict['idx'])}")
#     print(f"n count: {len(n_dict['idx'])}")
#     print(f"a ratio: {a_count_ratio}")
#     print(f"n ratio: {n_count_ratio}")
#     print(f"non_overlapping: {non_overlapping}")
#     print(salience)
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(20,5))
        ax.plot(score, c="b", label="score")
        ax.plot(label, c="g", label="label")
        ax.hlines(n_dict["mean"], 0, label.shape[0], "b", label=f"normal_plane:{n_dict['mean']:.3f}")
        ax.hlines(a_dict["mean"], 0, label.shape[0], "r", label=f"anomaly_plane:{a_dict['mean']:.3f}")
        ax.hlines(0, 0, label.shape[0], "r", label=f"salience:{salience:.3f}")
        ax.hlines(0, 0, label.shape[0], "r", label=f"overlapping:{overlapping:.3f}")
        ax.scatter(n_dict["idx"], salient_score_n, c="g")
        ax.scatter(a_dict["idx"], salient_score_a, c="r")
        
        ax.fill_between(np.arange(len(score)), a_dict["mean"]-a_dict["std"], a_dict["mean"]+a_dict["std"], alpha=0.2, facecolor = "red")
        ax.fill_between(np.arange(len(score)), n_dict["mean"]-n_dict["std"], n_dict["mean"]+n_dict["std"], alpha=0.2, facecolor = "green")
        
        ax.legend()
        
        if fig_saving_path:
            ax.figure.savefig(fig_saving_path)
    return salience

import pickle
with open("./compute_delay/all_best_scores.pkl", "rb") as fr:
    res_dict = pickle.load(fr)
    
from common.evaluation import compute_delay
salience_dict = defaultdict(list)
for hash_id, model_id in list(zip(best_data["hash_id"], best_data["model"])):
    dataset = best_data.loc[(best_data["model"]==model_id) & (best_data["hash_id"]==hash_id), "dataset"].tolist()[0]
#     if dataset != "MSL": continue
#     if model_id != "omnianomaly": continue
    print(model_id, dataset, "begin")

    salience_list = []
    for subdataset, value_dict in list(res_dict[(hash_id, model_id)].items()):
        train_score = value_dict["train_score"]
        test_score = value_dict["test_score"]
        label = value_dict["label"]
        
        salience = compute_salience(test_score, label, plot=False, ax=None, fig_saving_path="")
        salience_list.append(salience)
    avg_saliencee = np.array(salience_list).mean()
    salience_dict[(model_id, dataset)] = avg_saliencee
    
    print(model_id, dataset, "done")
    

def reorder_df(df):
    predefine_order = ["KNN", "iforest", "LODA", "LOF", "PCA", "AutoEncoder", "lstm", "lstm_vae", "dagmm", "mad_gan", "mscred", "omnianomaly"]
    reorder_indice = []
    for m in predefine_order:
        reorder_indice.append(list(df.loc[df["Model"] == m].index)[0])
#     print(reorder_indice)
    return df.loc[reorder_indice]


# writer1 = pd.ExcelWriter('all_salience.xlsx', engine='xlsxwriter')


#     f1_df = reorder_df(pd.DataFrame(dataset_rows_f1[dataset], columns= ["Model", "EVT_f1", "EVT_f1_a", "Search_f1", "Search_f1_a"]))
#     delay_df = reorder_df(pd.DataFrame(dataset_rows_delay[dataset], columns= ["Model", "pdelay", "rdelay"]))
#     f1_df.to_excel(writer1, sheet_name=dataset, index=False)
#     delay_df.to_excel(writer2, sheet_name=dataset, index=False) 
df_dict= {}
model_list = ["KNN", "iforest", "LODA", "LOF", "PCA", "AutoEncoder", "lstm", "lstm_vae", "dagmm", "mad_gan", "mscred", "omnianomaly"]
for dataset in ["SMD","SMAP","MSL","WADI","SWAT"]:
    salience_list = []
    for model in model_list:
        salience_list.append(salience_dict[(model, dataset)]) 
    df_dict[dataset] = salience_list
final_df = pd.DataFrame(df_dict)
final_df["model"] = model_list
final_df.set_index("model", inplace=True)
final_df.to_csv("all_salience_computed.csv")