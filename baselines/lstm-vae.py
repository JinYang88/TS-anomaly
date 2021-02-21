import os
import sys

os.chdir("../")
sys.path.append("./")
import pandas as pd
import pickle
import tensorflow as tf

from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess
from IPython import embed
from common import data_preprocess
from common.dataloader import load_dataset
from common.sliding import WindowIterator
from common.utils import score2pred, iter_thresholds

dataset = "SMD"
subdataset = "machine-1-1"

data_dict = load_dataset(
    dataset,
    subdataset,
    "all",
)

data_dict["train"] = preprocess(
    data_dict["train"]
)  # return normalized df, check NaN values replacing it with 0
data_dict["test"] = preprocess(
    data_dict["test"]
)  # return normalized df, check NaN values replacing it with 0

window_dict = data_preprocess.generate_windows(
    data_dict,
    window_size=32,
    stride=5,
)

train = window_dict["train_windows"]
test = window_dict["test_windows"]

df_train = train.reshape(-1, *train.shape[-2:])
df_test = test.reshape(-1, *test.shape[-2:])


vae = LSTM_Var_Autoencoder(
    intermediate_dim=15, z_dim=3, n_dim=df_train.shape[-1], stateful=True
)  # default stateful = False

vae.fit(
    df_train,
    learning_rate=0.001,
    batch_size=64,
    num_epochs=20,
    opt=tf.train.AdamOptimizer,
    REG_LAMBDA=0.01,
    grad_clip_norm=10,
    optimizer_params=None,
    verbose=True,
)

x_reconstructed, recons_error = vae.reconstruct(
    df_test, get_error=True
)  # returns squared error

score = recons_error.mean(axis=-1)[:, -1]
anomaly_label = window_dict["test_labels"][:, -1]

f1_adjusted, theta, pred_adjusted, pred_raw = iter_thresholds(score, anomaly_label)
