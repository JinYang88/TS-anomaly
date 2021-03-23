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

dataset = "SMD"
subdataset = "machine-1-1"
window_size = 32
stride = 5
intermediate_dim = 64
z_dim = 3
stateful = True
learning_rate = 0.001
batch_size = 64
num_epochs = 1
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
        comp_hiddens=[32, 16, 2],
        comp_activation=tf.nn.tanh,
        est_hiddens=[80, 40],
        est_activation=tf.nn.tanh,
        est_dropout_ratio=0.25,
        minibatch_size=1024,
        epoch_size=100,
        learning_rate=0.0001,
        lambda1=0.1,
        lambda2=0.0001,
        normalize=True,
        random_seed=123,
    )

    # predict anomaly score
    dagmm.fit(x_train)
    anomaly_score = dagmm.predict_prob(x_test)
    anomaly_label = x_test_labels

    # print(anomaly_score.shape)
    # print(anomaly_label.shape)

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
