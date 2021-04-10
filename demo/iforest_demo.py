import sys
import logging
from alibi_detect.od import IForest

sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint

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

<<<<<<< HEAD
    # data preprocessing for MSCRED

=======
>>>>>>> 9e393b9592b2467b8bddc4b731edb552103fb2de
    od = IForest()

    od.fit(x_train)

    anomaly_score = od.score(x_test)

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
