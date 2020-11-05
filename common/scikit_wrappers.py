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
import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection
import utils
import networks
import sys

from common import triplet_loss
from IPython import embed

class TimeSeriesEncoder(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, window_size, nb_steps, lr, penalty, early_stopping,
                 encoder, params, in_channels, out_channels, device="cpu"):
        self.device = device
        self.architecture = ''
        self.batch_size = batch_size
        self.window_size = window_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
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
                print('Epoch: ', epochs + 1)
            for idx, batch in enumerate(train_iterator.loader):
                # batch: b x d x dim
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                if not varying:
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
                    print("Step: {}, loss: {:.3f}".format(i, loss.item()))
                if i >= self.nb_steps:
                    break
            epochs += 1
        return self


    def encode(self, bacher, X, batch_size=50):
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test_generator = bacher.get_iterator(X) 

        features = []
        self.encoder = self.encoder.eval()

        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    batch = batch.to(self.device)
                    features.append(self.encoder(batch))
            else:
                for batch in test_generator:
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

    def predict(self, X, batch_size=50):
        raise NotImplementedError("TBD")

    def score(self, X, y, batch_size=50):
        raise NotImplementedError("TBD")


class CausalCNNEncoder(TimeSeriesEncoder):
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, window_size=100, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, device="cpu"):

        super(CausalCNNEncoder, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size, window_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size, device),
            self.__encoder_params(in_channels, channels, depth, reduced_size,out_channels, kernel_size),
            in_channels, out_channels, device
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, device):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder = encoder.double().to(device)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    batch = batch.to(self.device)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    after_pool = after_pool.to(self.device)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1
            else:
                for batch in test_generator:
                    batch = batch.to(self.device)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    after_pool = after_pool.to(self.device)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'device': self.device,
            'gpu': self.gpu
        }

    def set_params(self, **kwargs):
        self.__init__(**kwargs)
        return self


class LSTMEncoder(TimeSeriesEncoder):
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, in_channels=1, device="cpu"):
        super(LSTMEncoder, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(device), {}, in_channels, 160, device
        )
        assert in_channels == 1
        self.architecture = 'LSTM'

    def __create_encoder(self, device):
        encoder = networks.lstm.LSTMEncoder()
        encoder.double()
        encoder = encoder.to(device)
        return encoder

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'in_channels': self.in_channels,
            'device': self.device
        }

    def set_params(self, **kwargs):
        self.__init__(**kwargs)
        return self
