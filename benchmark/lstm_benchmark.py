import sys

sys.path.append("../")

import os
import hashlib
import traceback
from common import data_preprocess
from common.dataloader import load_dataset
from common.batching import WindowIterator
from common.utils import print_to_json, update_from_nni_params, seed_everything, pprint
from networks.lstm import LSTM

import argparse
from common.config import subdatasets
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

seed_everything(2020)


# python lstm_benchmark.py --dataset MSL --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 128 --gpu 0
# python lstm_benchmark.py --dataset SMD --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 128 --gpu 0
# python lstm_benchmark.py --dataset SMD --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 64 --gpu 0
# python lstm_benchmark.py --dataset SMD --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 64 --gpu -1
# python lstm_benchmark.py --dataset WADI --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 64 --gpu -1
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")

parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--window_size", type=int, help="window_size")
parser.add_argument("--stride", type=int, help="stride")
parser.add_argument("--num_layers", type=int, help="num_layers")
parser.add_argument("--hidden_size", type=int, help="hidden_size")
parser.add_argument("--gpu", type=int, default=0, help="The gpu index, -1 for cpu")

args = vars(parser.parse_args())

model_name = "lstm"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

dataset = args["dataset"]
device = args["gpu"]
window_size = args["window_size"]
stride = args["stride"]
lr = args["lr"]
hidden_size = args["hidden_size"]
num_layers = args["num_layers"]


normalize = "minmax"
nb_epoch = 1000
patience = 5
dropout = 0
batch_size = 1024
prediction_length = 1
prediction_dims = []

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            save_path = os.path.join("./savd_dir_lstm", hash_id, subdataset)
            if subdataset != "machine-3-5": continue
            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(dataset, subdataset, "all", root_dir="../")

            pp = data_preprocess.preprocessor()
            data_dict = pp.normalize(data_dict, method=normalize)
            os.makedirs(save_path, exist_ok=True)
            pp.save(save_path)

            window_dict = data_preprocess.generate_windows(
                data_dict,
                window_size=window_size,
                stride=stride,
            )

            train_iterator = WindowIterator(
                window_dict["train_windows"], batch_size=batch_size, shuffle=True
            )
            test_iterator = WindowIterator(
                window_dict["test_windows"], batch_size=4096, shuffle=False
            )

            encoder = LSTM(
                in_channels=data_dict["dim"],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                window_size=window_size,
                prediction_length=prediction_length,
                prediction_dims=prediction_dims,
                patience=patience,
                save_path=save_path,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                lr=lr,
                device=device,
            )

            encoder.fit(
                train_iterator,
                test_iterator=test_iterator.loader,
                test_labels=window_dict["test_labels"],
            )

            encoder.load_encoder()
            records = encoder.predict_prob(test_iterator.loader)

            records_train = encoder.predict_prob(train_iterator.loader)

            train_anomaly_score = records_train["score"]
            anomaly_score = records["score"]
            anomaly_label = window_dict["test_labels"][:, -1]

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                {"train": train_anomaly_score, "test": anomaly_score},
                anomaly_label,
                encoder.time_tracker,
            )
        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )
