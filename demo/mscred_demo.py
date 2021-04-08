import sys
import logging

sys.path.append("../")

from networks.mscred.mscred import MSCRED
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint

dataset = "SMD"
subdataset = "machine-1-1"
device = "cpu"  # cuda:0, a string
step_max = 5
gap_time = 10
win_size = [10, 30, 60]  # sliding window size
in_channels_encoder = 3
in_channels_decoder = 256
save_path = "./mscred_data/"
matrix_path = "machine1-1"    # defined by user
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

    mscred = MSCRED(
        in_channels_encoder,
        in_channels_decoder,
        matrix_path,
        save_path,
        device,
        step_max,
        gap_time,
        win_size,
        learning_rate,
        epoch,
        thred_b,
    )

    mscred.data_preprocessing(data_dict)

    mscred.fit()

    anomaly_score, anomaly_label = mscred.predict_prob(x_test, x_test_labels)

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
