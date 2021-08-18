import sys
import logging
# from alibi_detect.od import IForest
from pyod.models.iforest import IForest

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
# python iforest_benchmark.py --dataset SMD --n_estimators 100

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SMD", help="dataset")
parser.add_argument("--n_estimators", type=int, default=100, help="n_estimators")
args = vars(parser.parse_args())

# parameters are got from the args
dataset = args["dataset"]
n_estimators = args["n_estimators"]

model_name = "iforest"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            time_tracker = {}
            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(dataset, subdataset, "all", root_dir="../")

            x_train = data_dict["train"]
            x_test = data_dict["test"]
            x_test_labels = data_dict["test_labels"]

            od = IForest(n_estimators=n_estimators)

            train_start = time.time()
            od.fit(x_train)
            train_end = time.time()

            test_start = time.time()
            anomaly_score = od.decision_function(x_test)
            test_end = time.time()

            anomaly_score_train = od.decision_function(x_train)

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
                {"test": anomaly_score, "train": anomaly_score_train},
                anomaly_label,
                time_tracker,
            )

        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )