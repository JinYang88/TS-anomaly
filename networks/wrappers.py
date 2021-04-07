# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import logging
import math
import os
import sys
from collections import defaultdict
from glob import glob

import joblib
import numpy
import time
import numpy as np
import sklearn
import sklearn.externals
import sklearn.model_selection
import sklearn.svm
import torch
from common.evaluation import iter_thresholds
from IPython import embed
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import networks


class TimeSeriesEncoder(torch.nn.Module):
    def __init__(
        self,
        save_path,
        batch_size,
        nb_epoch,
        lr,
        device="cpu",
        architecture="base",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.best_metric = -float("inf")
        self.time_tracker = {}
        self.model_save_file = os.path.join(save_path, f"{architecture}_model.ckpt")

    def compile(self):
        print("Compiling finished.")
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.001
        )
        self = self.to(self.device)

    def save_encoder(self):
        print("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_encoder(self, model_save_path=""):
        print("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(self.model_save_file, map_location=self.device))

    def fit(
        self,
        train_iterator,
        test_iterator=None,
        test_labels=None,
        percent=88,
        nb_epoch_per_verbose=300,
        save_memory=False,
        monitor="AUC",
        patience=10,
        **kwargs,
    ):
        # Check if the given time series have unequal lengths
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        num_batches = len(train_iterator.loader)
        print("Start training for {} batches.".format(num_batches))
        train_start = time.time()
        # Encoder training
        while epochs < self.nb_epoch:
            running_loss = 0
            for idx, batch in enumerate(train_iterator.loader):
                # batch: b x d x dim
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                self.optimizer.zero_grad()
                loss = return_dict["loss"]
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / num_batches
            print("Epoch: {}, loss: {:.5f}".format(epochs + 1, avg_loss))
            self.__on_epoch_end(avg_loss, patience=patience)
            epochs += 1
        train_end = time.time()

        self.time_tracker["train"] = train_end - train_start
        return self

    def __on_epoch_end(self, monitor_value, patience):
        if monitor_value < self.best_metric:
            self.best_metric = monitor_value
            print("Saving model for performance: {:.3f}".format(monitor_value))
            self.save_encoder()
            self.worse_count = 0
        else:
            self.worse_count += 1
        if self.worse_count >= patience:
            return True
        return False

    def encode(self, iterator):
        # Check if the given time series have unequal lengths
        save_dict = defaultdict(list)
        self = self.eval()

        used_keys = ["recst", "y", "diff"]
        with torch.no_grad():
            for batch in iterator:
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                for k in used_keys:
                    save_dict[k].append(return_dict[k])
        self = self.train()
        return {k: torch.cat(v) for k, v in save_dict.items()}

    def encode_windows(self, windows):
        # window: n_batch x dim x time
        windows = torch.Tensor(windows)
        if len(windows.size()) == 2:
            windows = windows.unsqueeze(0)
        return self(windows)

    def predict(self, X):
        raise NotImplementedError("TBD")

    def score(self, iterator, anomaly_label, percent=88):
        print("Evaluating")
        self = self.eval()
        test_start = time.time()
        with torch.no_grad():
            score_list = []
            for batch in iterator:
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                score = (
                    # average all dimension
                    return_dict["score"]
                    .mean(dim=-1)
                    .sigmoid()  # b x prediction_length
                )
                # mean all timestamp
                score_list.append(score.mean(dim=-1))
        test_end = time.time()
        self.time_tracker["test"] = test_end - test_start
        anomaly_label = anomaly_label[:, -1]  # actually predict the last window
        score_list = torch.cat(score_list, dim=0).cpu().numpy()
        auc = roc_auc_score(anomaly_label, score_list)

        f1_adjusted, theta, pred_adjusted, pred_raw = iter_thresholds(
            score_list, anomaly_label
        )
        ps_adjusted = precision_score(pred_adjusted, anomaly_label)
        rc_adjusted = recall_score(pred_adjusted, anomaly_label)

        f1_raw = f1_score(pred_raw, anomaly_label)
        ps_raw = precision_score(pred_raw, anomaly_label)
        rc_raw = recall_score(pred_raw, anomaly_label)

        self = self.train()
        return {
            "score": score_list,
            "pred_raw": pred_raw,
            "pred": pred_adjusted,
            "anomaly_label": anomaly_label,
            "theta": theta,
            "AUC": auc,
            "F1": f1_raw,
            "PS": ps_raw,
            "RC": rc_raw,
            "F1_adj": f1_adjusted,
            "PS_adj": ps_adjusted,
            "RC_adj": rc_adjusted,
        }