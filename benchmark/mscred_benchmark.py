import os
import sys
import argparse
import hashlib
import traceback

sys.path.append("../")

from networks.mscred.mscred import MSCRED
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from common.config import subdatasets
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

# python mscred_benchmark.py --dataset SMD --lr 0.001 --in_channels_encoder 3 --in_channels_decoder 256 --hidden_size 64 --num_epochs 1 --gpu 0

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--in_channels_encoder", type=int, help="in_channels_encoder")
parser.add_argument("--in_channels_decoder", type=int, help="in_channels_decoder")
parser.add_argument("--hidden_size", type=int, help="hidden_size")
parser.add_argument("--num_epochs", type=int, help="num_epochs")
parser.add_argument("--gpu", type=int, default=0, help="The gpu index, -1 for cpu")

args = vars(parser.parse_args())


model_name = "mscred"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

dataset = args["dataset"]
device = args["gpu"]  # cuda:0, a string
in_channels_encoder = args["in_channels_encoder"]  # 3
in_channels_decoder = args["in_channels_decoder"]  # 256
learning_rate = args["lr"]
epoch = args["num_epochs"]

win_size = [10, 20, 30]  # sliding window size
step_max = 5
gap_time = 10
thred_b = 0.005

if __name__ == "__main__":
    for subdataset in subdatasets[dataset][0:2]:
        try:
            save_path = os.path.join("./mscred_data/", subdataset)

            # load dataset
            data_dict = load_dataset(dataset, subdataset, "all", nrows=100)

            x_train = data_dict["train"]
            x_test = data_dict["test"]
            x_test_labels = data_dict["test_labels"]

            # data preprocessing for MSCRED

            mscred = MSCRED(
                in_channels_encoder,
                in_channels_decoder,
                save_path,
                device,
                step_max,
                gap_time,
                win_size,
                learning_rate,
                epoch,
                thred_b,
            )

            mscred.fit(data_dict)

            anomaly_score, anomaly_label = mscred.predict_prob(x_test, x_test_labels)

            print(anomaly_score.shape)
            print(anomaly_label.shape)
            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                anomaly_score,
                anomaly_label,
                mscred.time_tracker,
            )

        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )