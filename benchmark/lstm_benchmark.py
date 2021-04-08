import os
import sys

sys.path.append("../")
from common import data_preprocess
from common.config import (
    initialize_config,
    parse_arguments,
    set_logger,
    subdatasets,
    get_trial_id,
)
from common.dataloader import load_dataset
from common.batching import WindowIterator
from common.utils import print_to_json, update_from_nni_params, seed_everything, pprint
from networks.lstm import LSTM
from common.evaluation import evaluator, store_output

seed_everything(2020)

dataset = "SMD"
normalize = "minmax"
save_path = "./savd_dir"
batch_size = 1024
device = "cuda:0"
window_size = 32
stride = 5
nb_epoch = 1000
patience = 5

lr = 0.001
hidden_size = 64
num_layers = 1
dropout = 0
prediction_length = 1
prediction_dims = []
iterate_threshold = False
point_adjustment = False

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        print(f"Running on {subdataset} of {dataset}")
        data_dict = load_dataset(dataset, subdataset, "all", root_dir="../")

        pp = data_preprocess.preprocessor()
        data_dict = pp.normalize(data_dict, method=normalize)
        os.makedirs(save_path, exist_ok=True)
        pp.save(save_path)

        window_dict = data_preprocess.generate_windows(
            data_dict,
            window_size=window_size,
            stride=stride,
        )

        train_iterator = WindowIterator(
            window_dict["train_windows"], batch_size=batch_size, shuffle=True
        )
        test_iterator = WindowIterator(
            window_dict["test_windows"], batch_size=4096, shuffle=False
        )

        print("Proceeding using {}...".format(device))

        encoder = LSTM(
            in_channels=data_dict["dim"],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            window_size=window_size,
            prediction_length=prediction_length,
            prediction_dims=prediction_dims,
            patience=patience,
            save_path=save_path,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            lr=lr,
            device=device,
        )

        encoder.fit(
            train_iterator,
            test_iterator=test_iterator.loader,
            test_labels=window_dict["test_labels"],
        )

        encoder.load_encoder()
        records = encoder.score(test_iterator.loader, window_dict["test_labels"])

        anomaly_score = records["score"]
        anomaly_label = records["anomaly_label"]

        print(anomaly_score.shape, anomaly_label.shape)

        # eva = evaluator(
        #     ["auc", "f1", "pc", "rc"],
        #     anomaly_score,
        #     anomaly_label,
        #     iterate_threshold=iterate_threshold,
        #     iterate_metric="f1",
        #     point_adjustment=point_adjustment,
        # )
        # eval_results = eva.compute_metrics()

        # pprint(eval_results)
        store_output(
            "./output",
            dataset,
            "lstm",
            subdataset,
            anomaly_score,
            anomaly_label,
        )
