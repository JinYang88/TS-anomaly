import os
import sys
import logging

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
sys.path.append("./")
from networks.mscred.matrix_generator import *
from networks.mscred.mscred import MSCRED
from networks.mscred.utils import *
from common.data_preprocess import generate_windows, preprocessor
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from IPython import embed

dataset = "SMD"
subdataset = "machine-1-1"
device = torch.device("cuda:0")
step_max = 5
gap_time = 10
win_size = [10, 30, 60]  # sliding window size
in_channels_encoder = 3
in_channels_decoder = 256
save_path = './mscred_data/'
learning_rate = 0.0002
epoch = 1
thred_b = 0.005
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
    )

    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    # data preprocessing for MSCRED

    mscred = MSCRED(in_channels_encoder, in_channels_decoder, data_dict, subdataset, x_train, x_test, x_test_labels,
                    save_path, step_max, gap_time, win_size, learning_rate, epoch, thred_b)

    mscred.data_preprocessing()

    mscred.fit()

    anomaly_score, anomaly_label = mscred.predict()

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
