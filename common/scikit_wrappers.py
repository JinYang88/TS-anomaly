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

from common import triplet_loss
from IPython import embed

class TimeSeriesEncoder(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,batch_size, nb_steps, lr, encoder, device="cpu", **kwargs):
        self.device = device
        self.architecture = ''
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.encoder = encoder
        self.loss = triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty, device)
        self.loss_varying = triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty, device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_encoder(self, prefix_file):
        try:
            torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )
        except:
            torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth',
            _use_new_zipfile_serialization=False 
        )

    def load_encoder(self, prefix_file):
        self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=self.device)
                )

    def fit(self, train_iterator, save_memory=False, verbose=False):
        # Check if the given time series have unequal lengths
        varying = False
        train = train_iterator.fetch_windows()
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                logging.info('Epoch: {}'.format(epochs + 1))
            for idx, batch in enumerate(train_iterator.loader):
                # batch: b x d x dim
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                if not varying:
                    logging.info("computing loss..")
                    loss = self.loss(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                loss.backward()
                self.optimizer.step()
                i += 1
                if i % 10 == 0:
                    logging.info("Step: {}, loss: {:.3f}".format(i, loss.item()))
                if i >= self.nb_steps:
                    break
            epochs += 1
        return self


    def encode(self, iterator):
        # Check if the given time series have unequal lengths
        varying = False
        features = []
        self.encoder = self.encoder.eval()

        with torch.no_grad():
            if not varying:
                for batch in iterator:
                    batch = batch.to(self.device)
                    features.append(self.encoder(batch))
            else:
                for batch in iterator:
                    batch = batch.to(self.device)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features.append(self.encoder(batch[:, :, :length]))

        features = torch.cat(features)
        self.encoder = self.encoder.train()
        return features

    def encode_windows(self, windows):
        # window: n_batch x dim x time
        windows = torch.Tensor(windows).double()
        if len(windows.size()) == 2:
            windows = windows.unsqueeze(0)
        return self.encoder(windows)

    def predict(self, X):
        raise NotImplementedError("TBD")

    def score(self, X, y):
        raise NotImplementedError("TBD")


class CausalCNNEncoder(TimeSeriesEncoder):
    def __init__(self, in_channels=1, channels=10, depth=1,reduced_size=10, out_channels=10, kernel_size=4, device="cpu", **kwargs):
        super(CausalCNNEncoder, self).__init__(
            encoder=self.__create_encoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size, device, **kwargs), device=device, **kwargs)
        self.architecture = 'CausalCNN'
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