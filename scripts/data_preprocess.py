import ast
import csv
import os
import sys
from pickle import dump

import numpy as np


from IPython import embed


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(
        os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb"
    ) as file:
        dump(temp, file)


def load_data(dataset):
    if dataset == "SMD":
        dataset_folder = "../datasets/anomaly/SMD"
        output_folder = "../datasets/anomaly/SMD/pkls"
        os.makedirs(output_folder, exist_ok=True)
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save("train", filename, filename.strip(".txt"), dataset_folder)
                load_and_save("test", filename, filename.strip(".txt"), dataset_folder)
                load_and_save(
                    "test_label", filename, filename.strip(".txt"), dataset_folder
                )
    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "../datasets/anomaly/SMAP-MSL"
        output_folder = "../datasets/anomaly/SMAP-MSL/processed_" + dataset
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        # data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]

        data_info = [row for row in res if row[1] == dataset]

        for row in data_info:
            subdataset = row[0]
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True

            train = np.load(os.path.join(dataset_folder, "train", subdataset + ".npy"))
            test = np.load(os.path.join(dataset_folder, "test", subdataset + ".npy"))

            print(
                subdataset,
                "test_label",
                label.shape,
                round(label.sum() / len(label), 2),
            )

            with open(
                os.path.join(output_folder, subdataset + "_" + "test_label" + ".pkl"),
                "wb",
            ) as fw:
                dump(label, fw)
            with open(
                os.path.join(output_folder, subdataset + "_train.pkl"),
                "wb",
            ) as fw:
                dump(train, fw)
            with open(
                os.path.join(output_folder, subdataset + "_test.pkl"),
                "wb",
            ) as fw:
                dump(test, fw)


if __name__ == "__main__":
    datasets = ["SMD", "SMAP", "MSL"]
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print(
            """
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
        """
        )
