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

import joblib
import logging
import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection
import networks
import sys
import os
from glob import glob
from common import triplet_loss
from common.utils import adjust_predicts
from IPython import embed
from sklearn.metrics import f1_score, precision_score, recall_score

class TimeSeriesEncoder(torch.nn.Module):
    def __init__(self, save_path, trial_id, compared_length, nb_random_samples, negative_penalty, batch_size, nb_steps, lr, architecture="BaseEncoder", device="cpu", **kwargs):
        super().__init__()
        self.architecture = architecture
        self.save_path = save_path
        self.trial_id = trial_id
        self.model_save_file = os.path.join(self.save_path, "{}_{}.pth".format(self.architecture, self.trial_id))

        self.device = device
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
    
    def compile(self):
        logging.info("Compiling finished.")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self = self.to(self.device)

    def save_encoder(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
            self.state_dict(),
            self.model_save_file,
            _use_new_zipfile_serialization=False 
        )
        except:
            torch.save(
            self.state_dict(),
            self.model_save_file
        )

    def load_encoder(self, model_save_path=""):
        if model_save_path:
            model_save_file = glob(os.path.join(model_save_path, "*.pth"))[0]
        else:
            model_save_file = self.model_save_file
        logging.info("Loading model from {}".format(model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def fit(self, train_iterator, test_iterator=None, test_labels=None, percent=88, nb_steps_per_verbose=300, save_memory=False, **kwargs):
        # Check if the given time series have unequal lengths
        train = train_iterator.fetch_windows()
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        logging.info("Start training for {} steps.".format(self.nb_steps))
        # Encoder training
        while epochs < self.nb_steps:
            logging.info('Epoch: {}'.format(epochs + 1))
            for idx, batch in enumerate(train_iterator.loader):
                # batch: b x d x dim
                batch = batch.to(self.device)
                return_dict = self(batch)
                self.optimizer.zero_grad()
                loss = return_dict["loss"]
                loss.backward()
                self.optimizer.step()
            self.score(test_iterator, test_labels, percent)
            epochs += 1
        return self

    def encode(self, iterator):
        # Check if the given time series have unequal lengths
        features = []
        self = self.eval()

        with torch.no_grad():
            for batch in iterator:
                batch = batch.to(self.device)
                return_dict = self(batch)
                features.append(return_dict["recst"])
        features = torch.cat(features)
        self = self.train()
        return features

    def encode_windows(self, windows):
        # window: n_batch x dim x time
        windows = torch.Tensor(windows).double()
        if len(windows.size()) == 2:
            windows = windows.unsqueeze(0)
        return self(windows)

    def predict(self, X):
        raise NotImplementedError("TBD")

    def score(self, iterator, anomaly_label, percent=88):
        logging.info("Evaluating ")
        self = self.eval()
        anomaly_label = anomaly_label[:, -1] # actually predict the last window
        with torch.no_grad():
            diff_list = []
            for batch in iterator:
                batch = batch.to(self.device)
                return_dict = self(batch)
                # diff = return_dict["diff"].max(dim=-1)[0] # chose the most anomaous ts
                diff = return_dict["diff"].mean(dim=-1) # chose the most anomaous ts
                diff_list.append(diff)

        diff_list = torch.cat(diff_list)
        pred = adjust_predicts(diff_list.cpu().numpy(), anomaly_label, percent)
        
        f1 = f1_score(pred, anomaly_label)
        ps = precision_score(pred, anomaly_label)
        rc = recall_score(pred, anomaly_label)

        logging.info("F1: {:.3f}, PS: {:.3f}, RC:{:.3f}".format(f1, ps, rc))
        self = self.train()


class CausalCNNEncoder(TimeSeriesEncoder):
    def __init__(self, in_channels=1, channels=10, depth=1,reduced_size=10, out_channels=10, kernel_size=4, device="cpu", **kwargs):
        super(CausalCNNEncoder, self).__init__(
            architecture="CausalCNN",
            encoder=self.__create_encoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size, device, **kwargs), device=device, **kwargs)
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, device, **kwargs):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size, **kwargs)
        encoder = encoder.double().to(device)
        return encoder

    def set_params(self, **kwargs):
        self.__init__(**kwargs)
        return self