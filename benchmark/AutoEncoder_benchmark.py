import sys
import logging
from pyod.models.auto_encoder import AutoEncoder

sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint

## import the following for benchmarking
import time
import hashlib
import traceback
import argparse
from common.config import subdatasets
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

# write example command here
# python AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 64 32 32 64 --batch_size 32 --epochs 100 --l2_regularizer 0.1
# input neuron layer size like: 64 32 32 64

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--hidden_neurons", nargs="+", type=int, help="hidden_neurons")
parser.add_argument("--batch_size", type=int, help="batch_size")
parser.add_argument("--epochs", type=int, help="epochs")
parser.add_argument("--l2_regularizer", type=float, help="l2_regularizer")
args = vars(parser.parse_args())

# parameters are got from the args
dataset = args["dataset"]
hidden_neurons = args["hidden_neurons"]
batch_size = args["batch_size"]
epochs = args["epochs"]
l2_regularizer = args["l2_regularizer"]

model_name = "AutoEncoder"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            time_tracker = {}
            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(dataset, subdataset)

            x_train = data_dict["train"]
            x_test = data_dict["test"]
            x_test_labels = data_dict["test_labels"]

            od = AutoEncoder(
                hidden_neurons=hidden_neurons,
                batch_size=batch_size,
                epochs=epochs,
                l2_regularizer=l2_regularizer,
                verbose=0,
            )

            train_start = time.time()
            od.fit(x_train)
            train_end = time.time()

            test_start = time.time()
            anomaly_score = od.decision_function(x_test)
            test_end = time.time()

            time_tracker = {
                "train": train_end - train_start,
                "test": test_end - test_start,
            }

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
                time_tracker,
            )

        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )