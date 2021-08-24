import sys
import logging

sys.path.append("../")

from networks.loda_uni import LODAUni

from common.dataloader import load_dataset
from common.evaluation import evaluator


# import the following for benchmarking
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
# python 15_LODA_uni_benchmark.py --dataset SMD --n_bins 2 --anomaly_threshold 0.2 --anomaly_ts_num 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SMD", help="dataset")
parser.add_argument("--n_bins", type=int,
                    default=2, help="n_bins")
parser.add_argument("--anomaly_threshold", type=float,
                    default=0.2, help="anomaly_threshold")
parser.add_argument("--anomaly_ts_num", type=float,
                    default=0.5, help="anomaly_ts_num")
args = vars(parser.parse_args())

# parameters are got from the args
dataset = args["dataset"]
n_bins = args["n_bins"]
anomaly_threshold = args["anomaly_threshold"]
anomaly_ts_num = args["anomaly_ts_num"]

model_name = "loda_univariate"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            time_tracker = {}
            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(
                dataset, subdataset, "all", root_dir="../")

            x_train = data_dict["train"]
            x_test = data_dict["test"]
            x_test_labels = data_dict["test_labels"]

            # data preprocessing for MSCRED
            od = LODAUni(anomaly_ts_num=anomaly_ts_num,
                         anomaly_threshold=anomaly_threshold)

            train_start = time.time()
            od.fit(x_train, n_bins=n_bins)
            train_end = time.time()

            test_start = time.time()
            anomaly_score = od.predict(x_test)
            test_end = time.time()

            anomaly_score_train = od.predict(x_train)

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
