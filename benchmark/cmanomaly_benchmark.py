import sys

sys.path.append("../")

import os
import hashlib
import traceback
from common import data_preprocess
from common.dataloader import load_dataset
from common.batching import TokenDataset
from common.utils import seed_everything
from common.vocab import Vocab
from networks.cmanomaly import CMAnomaly

import argparse
from common.config import subdatasets
from common.evaluation import (
    store_benchmarking_results,
    evaluate_benchmarking_folder,
)

seed_everything(2020)


# python cmanomaly_benchmark.py --dataset SMD --lr 0.001 --window_size 64 --stride 5 --embedding_dim 16 --nbins 10 --gpu 0

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset used")

parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--window_size", type=int, help="window_size")
parser.add_argument("--stride", type=int, help="stride")
parser.add_argument("--embedding_dim", type=int, help="embedding_dim")
parser.add_argument("--nbins", type=int, help="nbins")
parser.add_argument("--gpu", type=int, default=0, help="The gpu index, -1 for cpu")

args = vars(parser.parse_args())

model_name = "CMAnomaly"  # change this name for different models
benchmarking_dir = "./benchmarking_results"
hash_id = hashlib.md5(
    str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
).hexdigest()[0:8]

dataset = args["dataset"]
device = args["gpu"]
window_size = args["window_size"]
stride = args["stride"]
nbins = args["nbins"]
lr = args["lr"]
embedding_dim = args["embedding_dim"]


normalize = "standard"
nb_epoch = 1000
patience = 3
dropout = 0
batch_size = 128
prediction_length = 1
prediction_dims = []

if __name__ == "__main__":
    for subdataset in subdatasets[dataset]:
        try:
            save_path = os.path.join("./savd_dir_lstm", hash_id, subdataset)

            print(f"Running on {subdataset} of {dataset}")
            data_dict = load_dataset(dataset, subdataset, "all", root_dir="../")

            pp = data_preprocess.preprocessor()
            # data_dict = pp.normalize(data_dict, method=normalize)

            ### make symbols and convert to numerical features
            # data_dict = pp.symbolize(data_dict)
            # uniform, quantile
            data_dict = pp.symbolize(data_dict, n_bins=nbins, strategy="uniform")
            vocab = Vocab()
            vocab.build_vocab(data_dict)
            data_dict = vocab.transform(data_dict)
            ### end

            os.makedirs(save_path, exist_ok=True)
            pp.save(save_path)

            window_dict = data_preprocess.generate_windows(
                data_dict,
                window_size=window_size,
                stride=stride,
            )

            train_iterator = TokenDataset(
                vocab, window_dict["train_windows"], batch_size=batch_size, shuffle=True
            )
            test_iterator = TokenDataset(
                vocab, window_dict["test_windows"], batch_size=512, shuffle=False
            )

            encoder = CMAnomaly(
                in_channels=data_dict["train"].shape[1],
                nb_classes=len(vocab.label2idx),
                window_size=window_size,
                vocab_size=vocab.vocab_size,
                embedding_dim=embedding_dim,
                dropout=dropout,
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
            records = encoder.predict_prob(test_iterator.loader)

            records_train = encoder.predict_prob(train_iterator.loader)

            train_anomaly_score = records_train["score"]
            anomaly_score = records["score"]
            anomaly_label = window_dict["test_labels"][:, -1]

            eval_folder = store_benchmarking_results(
                hash_id,
                benchmarking_dir,
                dataset,
                subdataset,
                args,
                model_name,
                {"train": train_anomaly_score, "test": anomaly_score},
                anomaly_label,
                encoder.time_tracker,
            )
        except Exception as e:
            print(f"Running on {subdataset} failed.")
            print(traceback.format_exc())

    average_monitor_metric = evaluate_benchmarking_folder(
        eval_folder, benchmarking_dir, hash_id, dataset, model_name
    )
