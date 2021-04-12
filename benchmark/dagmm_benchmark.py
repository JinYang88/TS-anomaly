import os
import sys

sys.path.append("../")
from networks.dagmm.dagmm import DAGMM
from common.data_preprocess import generate_windows, preprocessor
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
import tensorflow as tf

import hashlib
import traceback
import argparse
from common.config import subdatasets
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

# python dagmm_benchmark.py --dataset SMD --lr 0.0001 --dropout 0.25 --num_epochs 1 -ch 32 16 2 -eh 80 40

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")

parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--dropout", type=float, help="dropout")
parser.add_argument("--num_epochs", type=int, help="num_epochs")
parser.add_argument(
    "-ch", "--compression_hiddens", nargs="+", help="compression_hiddens", required=True
)  # 32 16 2
parser.add_argument(
    "-eh", "--estimation_hiddens", nargs="+", help="estimation_hiddens", required=True
)  # 80 40
args = vars(parser.parse_args())

model_name = "dagmm"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]


dataset = args["dataset"]
estimation_dropout_ratio = args["dropout"]
epoch = args["num_epochs"]  # 100
lr = args["lr"]
compression_hiddens = args["compression_hiddens"]
estimation_hiddens = args["estimation_hiddens"]

minibatch = 1024
normalize = True
random_seed = 123

if __name__ == "__main__":
    for subdataset in subdatasets[dataset][0:2]:
        try:
            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(dataset, subdataset, nrows=500)
            # preprocessing
            pp = preprocessor()
            data_dict = pp.normalize(data_dict)

            x_train = data_dict["train"]
            x_test = data_dict["test"]
            x_test_labels = data_dict["test_labels"]

            dagmm = DAGMM(
                comp_hiddens=compression_hiddens,
                comp_activation=tf.nn.tanh,
                est_hiddens=estimation_hiddens,
                est_activation=tf.nn.tanh,
                est_dropout_ratio=estimation_dropout_ratio,
                minibatch_size=minibatch,
                epoch_size=epoch,
                learning_rate=lr,
                lambda1=0.1,
                lambda2=0.0001,
                normalize=True,
                random_seed=random_seed,
            )

            # predict anomaly score
            dagmm.fit(x_train)
            anomaly_score = dagmm.predict_prob(x_test)
            anomaly_label = x_test_labels

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                anomaly_score,
                anomaly_label,
                dagmm.time_tracker,
            )

        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())
    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )