import sys
import logging
import numpy as np
sys.path.append("../")
from networks.threesigma import ThreeSigma
from common.dataloader import load_dataset
from common.utils import pprint
from common.evaluation import evaluator
from IPython import embed

dataset = "SMD"
subdataset = "machine-1-1"
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
    od = ThreeSigma()
    od.fit(x_train)

    # get outlier scores
    anomaly_score = od.predict(x_test)
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
