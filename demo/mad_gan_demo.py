import os
import sys
import json
import numpy as np
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

num_epochs = 5
window_size = 32
batch_size = 64
stride = 5
dataset = "SMD"
subdataset = "machine-1-1"
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
    "learning_rate": 0.05,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "D_rounds": 1,
    "G_rounds": 3,
    "E_rounds": 1,
    "shuffle": True,
    "eval_mul": False,
    "wrong_labels": False,
    "identifier": dataset + "_" + subdataset,
    "sub_id": dataset + "_" + subdataset,
    "dp": False,
    "l2norm_bound": 1e-05,
    "batches_per_lot": 1,
    "dp_sigma": 1e-05,
    "use_time": False,
    "num_generated_features": 38,
    "num_signals": 38,
}

if __name__ == "__main__":

    data_dict = load_dataset(dataset, subdataset, "all")

    pp = preprocessor()
    data_dict = pp.normalize(data_dict)
    settings["num_generated_features"] = data_dict["dim"]
    settings["num_signals"] = data_dict["dim"]

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

    test_labels = test_labels.reshape([test_labels.shape[0], test_labels.shape[1], 1])
    index = index.reshape([index.shape[0], index.shape[1], 1])

    settings["seq_step"] = 1

    anomaly_score, anomaly_label = gan_model.detect(
        test,
        settings,
        test_labels,
        index,
    )
    anomaly_score_train = gan_model.detect(train, settings)

    eva = evaluator(
        ["auc", "f1", "pc", "rc"],
        anomaly_score,
        anomaly_label,
        iterate_threshold=True,
        iterate_metric="f1",
        point_adjustment=True,
    )

    eval_results = eva.compute_metrics()

    pprint(eval_results)