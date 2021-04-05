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
device = torch.device("cpu")
in_channels_encoder = 3
in_channels_decoder = 256
save_path = "./mscred_data/"
learning_rate = 0.0002
epoch = 1
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

    data = np.concatenate((data_dict["train"], data_dict["test"]), axis=0)

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    # data preprocessing for MSCRED
    generate_signature_matrix_node(data, subdataset, save_path)
    generate_train_test_data(subdataset, x_train, x_test, save_path)

    mscred = MSCRED(in_channels_encoder, in_channels_decoder, device=device)

    dataLoader = load_data(subdataset, save_path)

    # 训练阶段
    optimizer = torch.optim.Adam(mscred.parameters(), lr=learning_rate)
    train(dataLoader["train"], mscred, optimizer, epochs=epoch, device=device)
    print("保存 %s 的模型" % subdataset)

    if not os.path.exists("./mscred_data/checkpoints"):
        os.makedirs("./mscred_data/checkpoints")

    torch.save(
        mscred.state_dict(), "./mscred_data/checkpoints/model-" + subdataset + ".pth"
    )

    # 测试阶段
    mscred.load_state_dict(
        torch.load("./mscred_data/checkpoints/model-" + subdataset + ".pth")
    )
    mscred.to(device)
    test(
        dataLoader["test"],
        mscred,
        subdataset,
        x_test,
        save_dir=save_path,
        device=device,
    )
    print("测试 %s 完成" % subdataset)

    anomaly_score = evaluate(subdataset, save_path)
    anomaly_label = x_test_labels[
        9 - len(x_test) % 10 : 9 - len(x_test) % 10 + len(anomaly_score)
    ]

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
