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
import nni
import numpy
import time
import numpy as np
import sklearn
import sklearn.externals
import sklearn.model_selection
import sklearn.svm
import torch
from common import triplet_loss
from common.utils import score2pred
from IPython import embed
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import networks


class TimeSeriesEncoder(torch.nn.Module):
    def __init__(
        self,
        save_path,
        trial_id,
        batch_size,
        nb_steps,
        lr,
        architecture="BaseEncoder",
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.architecture = architecture
        self.save_path = save_path
        self.trial_id = trial_id
        self.model_save_file = os.path.join(
            self.save_path,
            "{}_{}_{}.pth".format(
                self.architecture,
                kwargs.get("subdataset", kwargs["dataset"]),
                self.trial_id,
            ),
        )

        self.device = device
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.best_metric = -float("inf")
        self.time_tracker = {}

    def compile(self):
        logging.info("Compiling finished.")
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.001
        )
        self = self.to(self.device)

    def save_encoder(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_encoder(self, model_save_path=""):
        if model_save_path:
            model_save_file = glob(os.path.join(model_save_path, "*.pth"))[0]
        else:
            model_save_file = self.model_save_file
        logging.info("Loading model from {}".format(model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def fit(
        self,
        train_iterator,
        test_iterator=None,
        test_labels=None,
        percent=88,
        nb_steps_per_verbose=300,
        save_memory=False,
        monitor="AUC",
        patience=10,
        **kwargs
    ):
        # Check if the given time series have unequal lengths
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        num_batches = len(train_iterator.loader)
        logging.info("Start training for {} batches.".format(num_batches))
        train_start = time.time()
        # Encoder training
        while epochs < self.nb_steps:
            running_loss = 0
            for idx, batch in enumerate(train_iterator.loader):
                # batch: b x d x dim
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                self.optimizer.zero_grad()
                loss = return_dict["loss"]
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)

                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / num_batches
            logging.info("Epoch: {}, loss: {:.5f}".format(epochs + 1, avg_loss))
            # if test_labels is not None:
            #     eval_result = self.score(test_iterator, test_labels, percent)
            #     nni.report_intermediate_result(eval_result["AUC"])
            epochs += 1
            # stopping = self.__on_epoch_end(eval_result[monitor], patience + 1)
            # if stopping:
            #     logging.info(
            #         "Early stop at epoch={}, best={:.3f}".format(
            #             epochs, self.best_metric
            #         )
            #     )
            #     break
        train_end = time.time()

        self.time_tracker["train"] = train_end - train_start
        return self

    def __on_epoch_end(self, monitor_value, patience):
        if monitor_value > self.best_metric:
            self.best_metric = monitor_value
            logging.info("Saving model for performance: {:3f}".format(monitor_value))
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

    def iter_thresholds(self, score, label, adjust=True):
        best_f1 = -float("inf")
        best_theta = None
        best_adjust = None
        best_raw = None
        for anomaly_ratio in np.linspace(1e-3, 1, 500):
            info_save = {}
            adjusted_anomaly, raw_predict = score2pred(
                score, label, percent=100 * (1 - anomaly_ratio), adjust=adjust
            )

            f1 = f1_score(adjusted_anomaly, label)
            if f1 > best_f1:
                best_f1 = f1
                best_adjust = adjusted_anomaly
                best_raw = raw_predict
                best_theta = anomaly_ratio
        return best_f1, best_theta, best_adjust, best_raw

    def score(self, iterator, anomaly_label, percent=88):
        logging.info("Evaluating")
        self = self.eval()
        test_start = time.time()
        with torch.no_grad():
            score_list = []
            for batch in iterator:
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                score = (
                    # sum all dimension
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

        f1_adjusted, theta, pred_adjusted, pred_raw = self.iter_thresholds(
            score_list, anomaly_label, adjust=True
        )
        ps_adjusted = precision_score(pred_adjusted, anomaly_label)
        rc_adjusted = recall_score(pred_adjusted, anomaly_label)

        # f1_raw, theta, pred_raw = self.iter_thresholds(
        #     score_list, anomaly_label, adjust=False
        # )
        f1_raw = f1_score(pred_raw, anomaly_label)
        ps_raw = precision_score(pred_raw, anomaly_label)
        rc_raw = recall_score(pred_raw, anomaly_label)

        logging.info(
            "AUC: {:.3f}, F1: {:.3f}({:.3f}), PS: {:.3f}({:.3f}), RC:{:.3f}({:.3f})".format(
                auc,
                f1_raw,
                f1_adjusted,
                ps_raw,
                ps_adjusted,
                rc_raw,
                rc_adjusted,
            )
        )
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


class CausalCNNEncoder(TimeSeriesEncoder):
    def __init__(
        self,
        in_channels=1,
        channels=10,
        depth=1,
        reduced_size=10,
        out_channels=10,
        kernel_size=4,
        device="cpu",
        **kwargs
    ):
        super(CausalCNNEncoder, self).__init__(
            architecture="CausalCNN",
            encoder=self.__create_encoder(
                in_channels,
                channels,
                depth,
                reduced_size,
                out_channels,
                kernel_size,
                device,
                **kwargs
            ),
            device=device,
            **kwargs
        )
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(
        self,
        in_channels,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        device,
        **kwargs
    ):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels,
            channels,
            depth,
            reduced_size,
            out_channels,
            kernel_size,
            **kwargs
        )
        encoder = encoder.to(device)
        return encoder

    def set_params(self, **kwargs):
        self.__init__(**kwargs)
        return self
