import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
sys.path.append("./")
from networks.da_gmm.dagmm import DAGMM
from common.data_preprocess import generate_windows, preprocessor
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from IPython import embed
import tensorflow as tf

dataset = "SMAP"
subdataset = "A-1"
compression_hiddens = [32, 16, 2]
compression_activation = tf.nn.tanh
estimation_hiddens = [80, 40]
estimation_activation = tf.nn.tanh
estimation_dropout_ratio = 0.25
minibatch = 1024
epoch = 100
lr = 0.0001
lambdaone = 0.1
lambdatwo = 0.0001
normalize = True
random_seed = 123
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    # preprocessing
    pp = preprocessor()
    data_dict = pp.normalize(data_dict)

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    dagmm = DAGMM(
        comp_hiddens=compression_hiddens,
        comp_activation=compression_activation,
        est_hiddens=estimation_hiddens,
        est_activation=estimation_activation,
        est_dropout_ratio=estimation_dropout_ratio,
        minibatch_size=minibatch,
        epoch_size=epoch,
        learning_rate=lr,
        lambda1=lambdaone,
        lambda2=lambdatwo,
        normalize=normalize,
        random_seed=random_seed,
    )

    # predict anomaly score
    dagmm.fit(x_train)
    anomaly_score = dagmm.predict_prob(x_test)
    anomaly_label = x_test_labels

    # Make evaluation
    eva = evaluator(
        ["auc", "f1", "pc", "rc"],
        anomaly_score,
        anomaly_label,
        iterate_threshold=iterate_threshold,
        iterate_metric="f1",
        point_adjustment=point_adjustment,
    )
    eval_results = eva.compute_metrics()

    pprint(eval_results)
