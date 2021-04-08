import os
import copy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering


metric_func = {
    "f1": f1_score,
    "pc": precision_score,
    "rc": recall_score,
    "auc": roc_auc_score,
}


def store_output(
    root_dir, dataset, subdataset, model_name, anomaly_score, anomaly_label
):
    store_dir = os.path.join(root_dir, dataset, subdataset, model_name)
    os.makedirs(store_dir, exist_ok=True)

    np.savez(os.path.join(store_dir, "anomaly_score"), anomaly_score)
    np.savez(os.path.join(store_dir, "label"), anomaly_label)
    print("Store output of {} to {} done.".format(model_name, store_dir))


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
        self.pred_raw = None
        self.threshold = threshold
        self.__iterate_threshold = iterate_threshold
        self.__iterate_metric = iterate_metric
        self.__point_adjustment = point_adjustment

    def compute_pred(self):
        if self.threshold is not None:
            self.pred_raw = self.pred = (self.score >= self.threshold).astype(int)
        if self.__iterate_threshold:
            print("Iterating threshold.")
            best_metric, best_theta, best_adjust, best_raw = iter_thresholds(
                self.score,
                self.label,
                self.__iterate_metric,
                adjustment=self.__point_adjustment,
            )
            self.pred = best_adjust
            self.pred_raw = best_raw
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
                results["raw_" + metric] = metric_func[metric](
                    self.pred_raw, self.label
                )

        return results


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


def compute_support(score, label, dtype="normal"):
    if dtype == "normal":
        score_idx = np.arange(len(score))[(label == 0).astype(bool)]
    elif dtype == "anomaly":
        score_idx = np.arange(len(score))[(label == 1).astype(bool)]

    clusters = []
    dscore = score[score_idx]
    clustering = AgglomerativeClustering(affinity="l1", linkage="complete").fit(dscore)
    cluster_labels = clustering.labels_

    for label in range(len(set(cluster_labels))):
        clusters.append(dscore[cluster_labels == label])
    max_label = max(enumerate(clusters), key=lambda x: np.mean(x[1]))[0]

    max_cluster = clusters[max_label]
    std = np.std(max_cluster)
    mean = np.mean(max_cluster)
    original_idx = score_idx[cluster_labels == max_label]

    ## plot internal
    #     plot_x = np.arange(len(dscore))
    #     scatter_x = plot_x[cluster_labels==max_label]
    #     plt.figure()
    #     plt.plot(plot_x, dscore)
    #     plt.scatter(scatter_x, max_cluster, c="r")
    #     plt.hlines(mean, 0, cluster_labels.shape[0], "r", label=f"mean:{mean:.3f}")
    #     mean_arr = np.array([mean] * len(plot_x))
    #     plt.fill_between(plot_x, mean_arr-std, mean_arr+std, alpha=0.2, facecolor = "green")
    #     plt.show()

    return_dict = {"mean": mean, "std": std, "idx": original_idx}
    return return_dict


def compute_salience(score, label, plot=False, ax=None, fig_saving_path=""):
    print("Computing salience")
    total_indice = np.arange(len(score))
    score_n = score[~label.astype(bool)]
    score_a = score[label.astype(bool)]

    score_n_idx = total_indice[~label.astype(bool)]
    n_dict = compute_support(score, label, "normal")
    salient_score_n = score[n_dict["idx"]]

    score_a_idx = total_indice[label.astype(bool)]
    a_dict = compute_support(score, label, "anomaly")
    salient_score_a = score[a_dict["idx"]]

    a_lower = a_dict["mean"] - a_dict["std"]
    n_upper = n_dict["mean"] + n_dict["std"]

    # print(n_dict["mean"] - n_dict["std"], n_dict["mean"] + n_dict["std"])
    # print(a_dict["mean"] - a_dict["std"], a_dict["mean"] + a_dict["std"])

    overlapping = (n_upper - a_lower if n_upper >= a_lower else 0) / (
        2 * (min(a_dict["std"], n_dict["std"]))
    )
    non_overlapping = 1 - overlapping
    a_count_ratio = len(a_dict["idx"]) / (len(a_dict["idx"]) + len(n_dict["idx"]))
    salience = non_overlapping * (
        a_count_ratio * a_dict["mean"] - (1 - a_count_ratio) * n_dict["mean"]
    )

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(score, c="b", label="score")
        ax.plot(label, c="g", label="label")
        ax.hlines(
            n_dict["mean"],
            0,
            label.shape[0],
            "b",
            label=f"normal_plane:{n_dict['mean']:.3f}",
        )
        ax.hlines(
            a_dict["mean"],
            0,
            label.shape[0],
            "r",
            label=f"anomaly_plane:{a_dict['mean']:.3f}",
        )
        ax.hlines(0, 0, label.shape[0], "r", label=f"salience:{salience:.3f}")
        ax.hlines(0, 0, label.shape[0], "r", label=f"overlapping:{overlapping:.3f}")
        ax.scatter(n_dict["idx"], salient_score_n, c="g")
        ax.scatter(a_dict["idx"], salient_score_a, c="r")

        ax.fill_between(
            np.arange(len(score)),
            a_dict["mean"] - a_dict["std"],
            a_dict["mean"] + a_dict["std"],
            alpha=0.2,
            facecolor="red",
        )
        ax.fill_between(
            np.arange(len(score)),
            n_dict["mean"] - n_dict["std"],
            n_dict["mean"] + n_dict["std"],
            alpha=0.2,
            facecolor="green",
        )
        ax.legend()

        if fig_saving_path:
            ax.figure.savefig(fig_saving_path)
    return salience