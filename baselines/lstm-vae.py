import os
import sys

os.chdir("../")
sys.path.append("./")
import pandas as pd
import pickle
import tensorflow as tf
import time
import argparse
from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess
from IPython import embed
from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import score2pred, iter_thresholds
import traceback
from common.config import (
    subdatasets,
    get_trial_id,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def run(dataset, subdataset, window_size):
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )
    print("Loading {}-{}".format(dataset, subdataset))
    data_dict["train"] = preprocess(
        data_dict["train"]
    )  # return normalized df, check NaN values replacing it with 0
    data_dict["test"] = preprocess(
        data_dict["test"]
    )  # return normalized df, check NaN values replacing it with 0

    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=window_size,
        stride=5,
    )

    train = window_dict["train_windows"]
    test = window_dict["test_windows"]

    df_train = train.reshape(-1, *train.shape[-2:])
    df_test = test.reshape(-1, *test.shape[-2:])

    vae = LSTM_Var_Autoencoder(
        intermediate_dim=64, z_dim=3, n_dim=df_train.shape[-1], stateful=True
    )  # default stateful = False

    train_start = time.time()
    vae.fit(
        df_train,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=1,
        opt=tf.train.AdamOptimizer,
        REG_LAMBDA=0.01,
        grad_clip_norm=2,
        optimizer_params=None,
        verbose=True,
    )
    train_time = time.time() - train_start

    test_start = time.time()
    x_reconstructed, recons_error = vae.reconstruct(
        df_test, get_error=True
    )  # returns squared error
    test_time = time.time() - test_start

    score = recons_error.mean(axis=-1)[:, -1]
    anomaly_label = window_dict["test_labels"][:, -1]

    f1_adjusted, theta, pred_adjusted, pred_raw = iter_thresholds(score, anomaly_label)

    auc = roc_auc_score(anomaly_label, score)
    f1 = f1_score(pred_raw, anomaly_label)
    ps_adjusted = precision_score(pred_adjusted, anomaly_label)
    rc_adjusted = recall_score(pred_adjusted, anomaly_label)

    records = {
        "AUC": auc,
        "F1": f1,
        "F1_adj": f1_adjusted,
        "RC_adj": rc_adjusted,
        "PS_adj": ps_adjusted,
        "train_time": train_time,
        "test_time": test_time,
    }

    return records


if __name__ == "__main__":
    dataset = "SMD"
    parser = argparse.ArgumentParser(
        description="Anomaly detection repository for TS datasets"
    )
    parser.add_argument(
        "--dataset", type=str, metavar="D", default="SMD", help="dataset name"
    )
    parser.add_argument(
        "--subdataset", type=str, metavar="D", default="", help="dataset name"
    )
    parser.add_argument(
        "-ws",
        "--window_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    dataset = args.dataset
    subdataset = args.subdataset
    window_size = args.window_size

    detail_dir = "./details"
    os.makedirs(detail_dir, exist_ok=True)
    start_time = get_trial_id()
    records = []
    # run each subdataset
    for subdataset in subdatasets[dataset][0:1]:
        try:
            records.append(
                run(
                    dataset,
                    subdataset,
                    window_size,
                )
            )
        except:
            print("Run {} failed.".format(subdataset))
            print(traceback.format_exc())
    records = pd.DataFrame(records)
    records.to_csv(
        "./{}/{}-{}-all.csv".format(detail_dir, dataset, start_time),
        index=False,
    )
    log = "{}\t{}\t{}\t{}\tAUC-{:.4f}\tF1-{:.4f}\tF1_adj-{:.4f}\tPS_adj-{:.4f}\tRC_adj-{:.4f}\ttrain-{:.4f}s\ttest-{:.4f}s\n".format(
        start_time,
        "ws=" + str(window_size),
        "lstm-vae",
        dataset + "_all",
        records["AUC"].mean(),
        records["F1"].mean(),
        records["F1_adj"].mean(),
        records["PS_adj"].mean(),
        records["RC_adj"].mean(),
        records["train_time"].sum(),
        records["test_time"].sum(),
    )
    with open("./total_results.csv", "a+") as fw:
        fw.write(log)