import copy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


metric_func = {
    "f1": f1_score,
    "pc": precision_score,
    "rc": recall_score,
    "auc": roc_auc_score,
}


class evaluator:
    def __init__(
        self,
        metrics,
        score,
        label,
        threshold=None,
        iterate_threshold=True,
        iterate_metric="f1",
        point_adjustment=True,
    ):
        assert (
            iterate_threshold and iterate_metric
        ), "[iterate_threshold] and [iterate_metric] should both be non-empty."

        assert (
            threshold is not None or iterate_threshold is not None
        ), "At least one of [threshold] and [iterate_threshold] should be non-empty."

        self.metrics = metrics
        self.score = score
        self.label = label
        self.pred = None
        self.threshold = None
        self.__iterate_threshold = True
        self.__iterate_metric = "f1"
        self.__point_adjustment = True

    def compute_pred(self):
        if self.threshold is not None:
            self.pred = (self.score >= self.threshold).astype(int)
        if self.__iterate_threshold:
            best_metric, best_theta, best_adjust, best_raw = iter_thresholds(
                self.score,
                self.label,
                self.__iterate_metric,
                adjustment=self.__point_adjustment,
            )
            self.pred = best_adjust
        return self.pred

    def compute_metrics(self):
        if self.pred is None:
            self.compute_pred()
        results = {}
        for metric in self.metrics:
            if metric == "auc":
                results[metric] = metric_func[metric](self.label, self.score)
            else:
                results[metric] = metric_func[metric](self.pred, self.label)
        return results


def a_new_metric():
    """
    Suppose to be a better metric for anomaly detection
    """
    pass


def iter_thresholds(score, label, metric="f1", adjustment=False):
    best_metric = -float("inf")
    best_theta = None
    best_adjust = None
    best_raw = None
    adjusted_pred = None
    for anomaly_ratio in np.linspace(1e-3, 1, 500):
        threshold = np.percentile(score, 100 * (1 - anomaly_ratio))
        pred = (score >= threshold).astype(int)

        if adjustment:
            pred, adjusted_pred = point_adjustment(pred, label)
        else:
            adjusted_pred = pred

        current_value = metric_func[metric](adjusted_pred, label)
        if current_value > best_metric:
            best_metric = current_value
            best_adjust = adjusted_pred
            best_raw = pred
            best_theta = threshold
    return best_metric, best_theta, best_adjust, best_raw


def point_adjustment(pred, label):
    """
    Borrow from https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/eval_methods.py
    """

    adjusted_pred = copy.deepcopy(pred)

    anomaly_state = False
    anomaly_count = 0
    latency = 0
    for i in range(len(adjusted_pred)):
        if label[i] and adjusted_pred[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    if not adjusted_pred[j]:
                        adjusted_pred[j] = True
                        latency += 1
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = True
    return pred, adjusted_pred
