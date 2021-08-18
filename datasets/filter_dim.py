import os
import sys
import pickle
import numpy as np

sys.path.append("../")
from common.config import subdatasets


use_dim_num = 20
dataset = "SMD"  # modify this to HUAWEI


np.random.seed(2022)
datapath = os.path.join("anomaly", dataset, "processed")

target_dir = os.path.join("anomaly", dataset + "_{}".format(use_dim_num), "processed")
os.makedirs(target_dir, exist_ok=True)

with open("dim_filter_{}.txt".format(dataset), "w") as log_fw:
    for subdataset in subdatasets[dataset]:
        train = pickle.load(
            open(os.path.join(datapath, "{}_train.pkl".format(subdataset)), "rb")
        )
        test = pickle.load(
            open(os.path.join(datapath, "{}_test.pkl".format(subdataset)), "rb")
        )
        test_label = pickle.load(
            open(os.path.join(datapath, "{}_test_label.pkl".format(subdataset)), "rb")
        )
        dim_num = train.shape[1]
        assert dim_num >= use_dim_num, "dim_num should >= use_dim_num"
        use_dim_list = np.random.randint(low=0, high=dim_num, size=use_dim_num)
        log_fw.write("{}, {}\n".format(subdataset, use_dim_list))

        train_sample = train[:, use_dim_list]
        test_sample = test[:, use_dim_list]

        with open(
            os.path.join(target_dir, "{}_train.pkl".format(subdataset)), "wb"
        ) as fw:
            pickle.dump(train_sample, fw)
        with open(
            os.path.join(target_dir, "{}_test.pkl".format(subdataset)), "wb"
        ) as fw:
            pickle.dump(test_sample, fw)
        with open(
            os.path.join(target_dir, "{}_test_label.pkl".format(subdataset)), "wb"
        ) as fw:
            pickle.dump(test_label, fw)