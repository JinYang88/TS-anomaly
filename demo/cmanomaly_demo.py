import os
import sys

sys.path.append("../")
from common import data_preprocess
from common.dataloader import load_dataset
from common.batching import TokenDataset
from common.utils import seed_everything, pprint
from networks.cmanomaly import CMAnomaly
from common.evaluation import evaluator
from common.vocab import Vocab
from IPython import embed

seed_everything(2020)

dataset = "SMD"
subdataset = "machine-3-5"
normalize = "minmax"
save_path = "./savd_dir"
batch_size = 64
device = 0  # -1 for cpu, 0 for cuda:0
window_size = 64
stride = 5
nb_epoch = 100
patience = 5

lr = 0.01
hidden_size = 64
num_layers = 1
dropout = 0
embedding_dim = 16
prediction_length = 1
prediction_dims = [0]
iterate_threshold = True
point_adjustment = True

if __name__ == "__main__":
    data_dict = load_dataset(
        dataset,
        subdataset,
    )

    pp = data_preprocess.preprocessor()
    data_dict = pp.normalize(data_dict, method=normalize)

    ### make symbols and convert to numerical features
    data_dict = pp.symbolize(data_dict)
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

    print("Proceeding using {}...".format(device))

    encoder = CMAnomaly(
        in_channels=data_dict["train"].shape[1],
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

    anomaly_score = records["score"]
    anomaly_label = window_dict["test_labels"][:, -1]

    print(anomaly_score.shape, anomaly_label.shape)

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
