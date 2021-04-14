import torch.utils.data
import os
import math
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from common.utils import set_device
from glob import glob


def load_signature_data(save_dir):
    dataset = {}
    splits = ["train", "test"]
    shuffle = {"train": True, "test": False}
    train_data_path = os.path.join(save_dir, "matrix", "train_data", "*.npy")
    test_data_path = os.path.join(save_dir, "matrix", "test_data", "*.npy")
    train_file_list = glob(train_data_path)
    test_file_list = glob(test_data_path)
    train_file_list.sort(key=lambda x: int(os.path.basename(x)[11:-4]))
    test_file_list.sort(key=lambda x: int(os.path.basename(x)[10:-4]))
    train_data, test_data = [], []
    for obj in train_file_list:
        train_matrix = np.load(obj)
        # train_matrix = np.transpose(train_matrix, (0, 2, 3, 1))
        train_data.append(train_matrix)

    for obj in test_file_list:
        test_matrix = np.load(obj)
        # test_matrix = np.transpose(test_matrix, (0, 2, 3, 1))
        test_data.append(test_matrix)

    dataset["train"] = torch.from_numpy(np.array(train_data)).float()
    dataset["test"] = torch.from_numpy(np.array(test_data)).float()

    dataloader = {
        x: torch.utils.data.DataLoader(
            dataset=dataset[x], batch_size=1, shuffle=shuffle[x]
        )
        for x in splits
    }
    return dataloader


def train(DataLoader, model, Optimizer, epochs, device):
    model = model.to(device)
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0
        for x in tqdm(DataLoader):
            x = x.to(device)
            x = x.squeeze()
            l = torch.mean((model(x) - x[-1].unsqueeze(0)) ** 2)
            train_l_sum += l
            Optimizer.zero_grad()
            l.backward()
            Optimizer.step()
            n += 1

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epochs, train_l_sum / n))


def test(DataLoader, model, len_x_train, save_dir, gap_time, device):
    print("------Testing-------")
    index = math.ceil(len_x_train / gap_time)
    reconstructed_data_path = os.path.join(save_dir, "reconstructed_data")

    os.makedirs(reconstructed_data_path, exist_ok=True)

    with torch.no_grad():
        for x in DataLoader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x)
            path_temp = os.path.join(
                reconstructed_data_path, "reconstructed_data_" + str(index) + ".npy"
            )
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            index += 1


def evaluate(save_dir, thred_b, gap_time):
    reconstructed_data_path = os.path.join(save_dir, "reconstructed_data")

    test_data_path = os.path.join(save_dir, "matrix", "test_data")
    test_file_list = glob(os.path.join(test_data_path, "*.npy"))
    test_file_list.sort(key=lambda x: int(os.path.basename(x)[10:-4]))
    test_start = int(os.path.basename(test_file_list[0])[10:-4])
    test_end = int(os.path.basename(test_file_list[-1])[10:-4])
    test_anomaly_score = np.zeros((test_end - test_start + 1, 1))

    for m in range(test_start, test_end + 1):
        path_temp_1 = os.path.join(test_data_path, "test_data_" + str(m) + ".npy")
        gt_matrix_temp = np.load(path_temp_1)

        path_temp_2 = os.path.join(
            reconstructed_data_path, "reconstructed_data_" + str(m) + ".npy"
        )

        reconstructed_matrix_temp = np.load(path_temp_2)
        # reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp, [0, 3, 1, 2])
        # print(reconstructed_matrix_temp.shape)
        # first (short) duration scale for evaluation
        select_gt_matrix = np.array(gt_matrix_temp)[-1][0]  # get last step matrix

        select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]

        # compute number of broken element in residual matrix
        select_matrix_error = np.square(
            np.subtract(select_gt_matrix, select_reconstructed_matrix)
        )
        num_broken = len(select_matrix_error[select_matrix_error > thred_b])

        # print num_broken
        test_anomaly_score[m - test_start] = num_broken

    test_anomaly_score = test_anomaly_score.ravel()
    anomaly_score = []

    for m in range(len(test_anomaly_score)):
        for j in range(gap_time):
            anomaly_score.append(test_anomaly_score[m])

    return anomaly_score
