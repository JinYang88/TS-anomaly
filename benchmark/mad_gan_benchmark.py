import os
import sys
import json
import numpy as np
import traceback
import hashlib
from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("../")

from common.data_preprocess import (
    generate_windows,
    preprocessor,
    generate_windows_with_index,
)
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint


from networks.mad_gan.AD import myADclass
from networks.mad_gan.MADwrapper import MAD_GAN
from networks.mad_gan import utils
from networks.mad_gan import DR_discriminator
from networks.mad_gan import model

from common.config import subdatasets
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)
import argparse


# python mad_gan_benchmark.py --dataset SMD --window_size 32 --stride 5 --lr 0.005 --num_epochs 3


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")
parser.add_argument("--window_size", type=int, help="window_size")
parser.add_argument("--num_epochs", type=int, help="num_epochs")
parser.add_argument("--lr", type=float, help="learning_rate")
parser.add_argument("--stride", type=int, help="stride")
args = vars(parser.parse_args())

model_name = "mad_gan"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]


num_epochs = args["num_epochs"]
window_size = args["window_size"]
stride = args["stride"]
dataset = args["dataset"]
learning_rate = args["lr"]

# print(num_epochs)
# sys.exit()
batch_size = 32
save_dir = "./experiments"

settings = {
    "eval_an": False,
    "eval_single": False,
    "seq_length": window_size,
    "seq_step": stride,
    "normalise": False,
    "scale": 0.1,
    "freq_low": 1.0,
    "freq_high": 5.0,
    "amplitude_low": 0.1,
    "amplitude_high": 0.9,
    "multivariate_mnist": False,
    "full_mnist": False,
    "resample_rate_in_min": 15,
    "hidden_units_g": 100,
    "hidden_units_d": 100,
    "hidden_units_e": 100,
    "kappa": 1,
    "latent_dim": 15,
    "weight": 0.5,
    "degree": 1,
    "batch_mean": False,
    "learn_scale": False,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "D_rounds": 1,
    "G_rounds": 3,
    "E_rounds": 1,
    "shuffle": True,
    "eval_mul": False,
    "wrong_labels": False,
    "dp": False,
    "l2norm_bound": 1e-05,
    "batches_per_lot": 1,
    "dp_sigma": 1e-05,
    "use_time": False,
    "num_generated_features": 38,
    "num_signals": 38,
}

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            data_dict = load_dataset(dataset, subdataset, "all")

            pp = preprocessor()
            data_dict = pp.normalize(data_dict)
            settings["num_generated_features"] = data_dict["dim"]
            settings["num_signals"] = data_dict["dim"]
            settings["identifier"] = dataset + "_" + subdataset
            settings["sub_id"] = dataset + "_" + subdataset

            # generate sliding windows
            window_dict = generate_windows_with_index(
                data_dict, window_size=window_size, stride=stride
            )

            train = window_dict["train_windows"]
            test = window_dict["test_windows"]
            test_labels = window_dict["test_labels"]
            index = window_dict["index_windows"]

            labels = np.zeros([train.shape[0], train.shape[1], 1])

            gan_model = MAD_GAN(save_dir)
            gan_model.fit(train, labels, settings)

            test_labels = test_labels.reshape(
                [test_labels.shape[0], test_labels.shape[1], 1]
            )
            index = index.reshape([index.shape[0], index.shape[1], 1])

            anomaly_score, anomaly_label = gan_model.detect(
                test, test_labels, index, settings
            )

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                anomaly_score,
                anomaly_label,
                gan_model.time_tracker,
            )

        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )
