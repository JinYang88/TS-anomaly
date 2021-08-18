import os
import sys
import hashlib
import argparse
import traceback
import tensorflow as tf

sys.path.append("../")
from networks.lstm_vae import LSTM_Var_Autoencoder
from common.data_preprocess import generate_windows, preprocessor
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.config import subdatasets
from IPython import embed
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

#  python 6_lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 10

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--window_size", type=int, help="window_size")
parser.add_argument("--stride", type=int, help="stride")
parser.add_argument("--intermediate_dim", type=int, help="intermediate_dim")
parser.add_argument("--z_dim", type=int, help="z_dim")
parser.add_argument("--hidden_size", type=int, help="hidden_size")
parser.add_argument("--num_epochs", type=int, help="num_epochs")
args = vars(parser.parse_args())

model_name = "lstm_vae"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

dataset = args["dataset"]
window_size = args["window_size"]
stride = args["stride"]
intermediate_dim = args["intermediate_dim"]  # 64
z_dim = args["z_dim"]  # 3
learning_rate = args["lr"]
num_epochs = args["num_epochs"]

stateful = False
batch_size = 1024

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(dataset, subdataset, "all", root_dir="../")

            # preprocessing
            pp = preprocessor()
            data_dict = pp.normalize(data_dict)

            # generate sliding windows
            window_dict = generate_windows(
                data_dict, window_size=window_size, stride=stride
            )

            train = window_dict["train_windows"]
            test = window_dict["test_windows"]
            test_labels = window_dict["test_labels"]

            # reshape to fitin lstm_vae (not all models need this)
            df_train = train.reshape(-1, *train.shape[-2:])
            df_test = test.reshape(-1, *test.shape[-2:])

            vae = LSTM_Var_Autoencoder(
                intermediate_dim=intermediate_dim,
                z_dim=z_dim,
                n_dim=df_train.shape[-1],
                stateful=stateful,
            )

            # Training
            vae.fit(
                df_train,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                REG_LAMBDA=0.01,
                grad_clip_norm=2,
                optimizer_params=None,
                verbose=True,
            )

            # predict anomaly score for each window
            anomaly_score_train = vae.predict_prob(df_train).mean(axis=-1)[
                :, -1
            ]  # mean for all dims
            anomaly_score = vae.predict_prob(df_test).mean(axis=-1)[
                :, -1
            ]  # mean for all dims
            anomaly_label = window_dict["test_labels"][
                :, -1
            ]  # last point of each window

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                {"train": anomaly_score_train, "test": anomaly_score},
                anomaly_label,
                vae.time_tracker,
            )

        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())
    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )