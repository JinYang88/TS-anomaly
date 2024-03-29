from collections import defaultdict
import os
import sys
import copy
import json
import glob
from IPython.terminal.embed import embed
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime

metric_func = {
    "f1": f1_score,
    "pc": precision_score,
    "rc": recall_score,
    "auc": roc_auc_score,
}


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            {k: str(v) for k, v in obj.items()},
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def store_benchmarking_results(
    hash_id,
    benchmark_dir,
    dataset,
    subdataset,
    args,
    model_name,
    anomaly_score,
    anomaly_label,
    time_tracker,
):
    value_store_dir = os.path.join(
        benchmark_dir, model_name, hash_id, dataset, subdataset
    )
    os.makedirs(value_store_dir, exist_ok=True)
    np.savez(os.path.join(value_store_dir, "anomaly_score"), anomaly_score)
    np.savez(os.path.join(value_store_dir, "anomaly_label"), anomaly_label)

    json_pretty_dump(time_tracker, os.path.join(value_store_dir, "time.json"))

    param_store_dir = os.path.join(benchmark_dir, model_name, hash_id)

    param_store = {"cmd": "python {}".format(" ".join(sys.argv))}
    param_store.update(args)

    json_pretty_dump(param_store, os.path.join(param_store_dir, "params.json"))
    print("Store output of {} to {} done.".format(model_name, param_store_dir))
    return os.path.join(benchmark_dir, model_name, hash_id, dataset)


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


def iter_thresholds(
    score, label, metric="f1", adjustment=False, normalized=False, threshold=None
):
    best_metric = -float("inf")
    best_theta = None
    best_adjust = None
    best_raw = None
    adjusted_pred = None
    if threshold is not None:
        search_range = [0]
    else:
        search_range = np.linspace(1e-3, 1, 100)

    best_set = []
    for trial in ["higher", "less"]:
        for anomaly_ratio in search_range:

            if sum(np.unique(score)) == 1 or sum(np.unique(score)) == 0:
                pred = score
                theta = None
            else:
                if threshold is None:
                    if not normalized:
                        theta = np.percentile(score, 100 * (1 - anomaly_ratio))
                    else:
                        theta = anomaly_ratio
                else:
                    theta = threshold

                if trial == "higher":
                    pred = (score > theta).astype(int)
                elif trial == "less":
                    pred = (score < theta).astype(int)

            if adjustment:
                pred, adjusted_pred = point_adjustment(pred, label)
            else:
                adjusted_pred = pred

            current_value = metric_func[metric](adjusted_pred, label)

            if current_value > best_metric:
                best_metric = current_value
                best_adjust = adjusted_pred
                best_raw = pred
                best_theta = theta
        best_set.append((best_metric, best_theta, best_adjust, best_raw))

    return max(best_set, key=lambda x: x[0])


def point_adjustment(pred, label):
    """
    Borrow from https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/eval_methods.py
    """
    raw_pred = copy.deepcopy(pred)
    actual = label == 1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(pred)):
        if actual[i] and pred[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not pred[j]:
                        pred[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            pred[i] = True
    return raw_pred, pred

    # anomaly_state = False
    # anomaly_count = 0
    # latency = 0
    # for i in range(len(adjusted_pred)):
    #     if label[i] and adjusted_pred[i] and not anomaly_state:
    #         anomaly_state = True
    #         anomaly_count += 1
    #         for j in range(i, 0, -1):
    #             if not label[j]:
    #                 break
    #             else:
    #                 if not adjusted_pred[j]:
    #                     adjusted_pred[j] = True
    #                     latency += 1
    #     elif not label[i]:
    #         anomaly_state = False
    #     if anomaly_state:
    #         adjusted_pred[i] = True
    # return pred, adjusted_pred


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


def evaluate_benchmarking_folder(
    folder,
    benchmarking_dir,
    hash_id,
    dataset,
    model_name,
    adjustment=True,
):
    concerned_metrics = [
        "train_time",
        "test_time",
        "raw_PC",
        "raw_RC",
        "raw_F1",
        "adj_PC",
        "adj_RC",
        "adj_F1",
    ]
    folder_count = 0
    foldernames = []
    metric_values_dict = defaultdict(list)
    pred_results_all = defaultdict(list)
    for subfolder in glob.glob(os.path.join(folder, "*")):
        folder_name = os.path.basename(subfolder)
        foldernames.append(folder_name)
        print("Evaluating {}".format(folder_name))
        anomaly_score = np.load(
            os.path.join(subfolder, "anomaly_score.npz"), allow_pickle=True
        )["arr_0"].item()["test"]
        anomaly_score_train = np.load(
            os.path.join(subfolder, "anomaly_score.npz"), allow_pickle=True
        )["arr_0"].item()["train"]
        anomaly_label = np.load(os.path.join(subfolder, "anomaly_label.npz"))[
            "arr_0"
        ].astype(int)
        with open(os.path.join(subfolder, "time.json")) as fr:
            time = json.load(fr)

        best_f1, best_theta, best_adjust_pred, best_raw_pred = iter_thresholds(
            anomaly_score, anomaly_label, metric="f1", adjustment=adjustment
        )

        pred_results_all["anomaly_adjust_pred"].append(best_adjust_pred)
        pred_results_all["anomaly_raw_pred"].append(best_raw_pred)
        pred_results_all["anomaly_score"].append(anomaly_score)
        pred_results_all["anomaly_label"].append(anomaly_label)
        pred_results_all["anomaly_score_train"].append(anomaly_score_train)

        try:
            auc = roc_auc_score(anomaly_label, anomaly_score)
        except ValueError as e:
            auc = 0
            print("All zero in anomaly label, set auc=0")

        adj_f1 = f1_score(anomaly_label, best_adjust_pred)
        adj_precision = precision_score(anomaly_label, best_adjust_pred)
        adj_recall = recall_score(anomaly_label, best_adjust_pred)

        raw_f1 = f1_score(anomaly_label, best_raw_pred)
        raw_precision = precision_score(anomaly_label, best_raw_pred)
        raw_recall = recall_score(anomaly_label, best_raw_pred)
        total_delay = compute_delay(anomaly_label, best_raw_pred)

        metric = {
            "auc": auc,
            "raw_F1": raw_f1,
            "raw_PC": raw_precision,
            "raw_RC": raw_recall,
            "adj_F1": adj_f1,
            "adj_PC": adj_precision,
            "adj_RC": adj_recall,
            "delay": total_delay,
            "train_time": time["train"],
            "test_time": time["test"],
        }
        for metric_name in concerned_metrics:
            metric_values_dict[metric_name].append(metric[metric_name])
        json_pretty_dump(metric, os.path.join(subfolder, "metrics.json"))
        folder_count += 1

    # concated_test_score = np.concatenate(pred_results_all["anomaly_score"])

    concated_raw_pred = np.concatenate(pred_results_all["anomaly_raw_pred"])
    concated_adjusted_pred = np.concatenate(pred_results_all["anomaly_adjust_pred"])
    concated_test_label = np.concatenate(pred_results_all["anomaly_label"])

    concacted_raw_f1 = f1_score(concated_test_label, concated_raw_pred)
    concacted_raw_precision = precision_score(concated_test_label, concated_raw_pred)
    concacted_raw_recall = recall_score(concated_test_label, concated_raw_pred)

    concacted_adj_f1 = f1_score(concated_test_label, concated_adjusted_pred)
    concacted_adj_precision = precision_score(
        concated_test_label, concated_adjusted_pred
    )
    concacted_adj_recall = recall_score(concated_test_label, concated_adjusted_pred)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(
        os.path.join(benchmarking_dir, f"{dataset}_{model_name}.txt"), "a+"
    ) as fw:
        params = " ".join(sys.argv)
        info = f"{current_time}\t{hash_id}\tcount:{folder_count}\t{params}\t"
        metric_str = []

        for metric_name in concerned_metrics:
            values = np.array(metric_values_dict[metric_name], dtype=float)
            mean, std = values.mean(), values.std()
            metric_str.append("{}: {:.3f} ({:.3f})".format(metric_name, mean, std))

        metric_str.append("con_adj_PC: {:.3f}".format(concacted_adj_precision))
        metric_str.append("con_adj_RC: {:.3f}".format(concacted_adj_recall))
        metric_str.append("con_adj_F1: {:.3f}".format(concacted_adj_f1))

        metric_str.append("con_PC: {:.3f}".format(concacted_raw_precision))
        metric_str.append("con_RC: {:.3f}".format(concacted_raw_recall))
        metric_str.append("con_F1: {:.3f}".format(concacted_raw_f1))

        metric_str = "\t".join(metric_str)
        info += metric_str + "\n"
        fw.write(info)

    metrics_per_datasets = pd.DataFrame(metric_values_dict, index=foldernames)
    metrics_per_datasets.to_csv(os.path.join(folder, "..", "metrics_per_datasets.csv"))
    print(info)


def compute_delay(label, pred):
    def onehot2interval(arr):
        result = []
        record = False
        for idx, item in enumerate(arr):
            if item == 1 and not record:
                start = idx
                record = True
            if item == 0 and record:
                end = idx  # not include the end point, like [a,b)
                record = False
                result.append((start, end))
        return result

    count = 0
    total_delay = 0
    pred = np.array(pred)
    label = np.array(label)
    for start, end in onehot2interval(label):
        pred_interval = pred[start:end]
        if pred_interval.sum() > 0:
            total_delay += np.where(pred_interval == 1)[0][0]
            count += 1
    return int(total_delay)


if __name__ == "__main__":
    eval_folder = "../benchmark/benchmarking_results/lstm/8a2c860e/SMD"
    print(evaluate_benchmarking_folder(eval_folder))

    # pred = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]
    # label = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    # compute_delay(pred, label)