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


import os
import json
import math
import torch
import numpy
import pandas
import argparse

from IPython import embed
from common.dataloader import load_SMD_dataset
from common import scikit_wrappers
from common.sliding import BatchSlidingWindow

def fit_hyperparameters(file, train, device,
                        save_memory=False):

    classifier = scikit_wrappers.CausalCNNEncoder()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['device'] = device
    classifier.set_params(**params)

    train_windows_batcher = BatchSlidingWindow(train.shape[0], window_size=params["window_size"], batch_size=params["batch_size"])

    return classifier.fit(
        train_windows_batcher, train, save_memory=save_memory, verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Anomaly detection repository for TS datasets')

    parser.add_argument('--dataset', type=str, metavar='D', default="machine-1-1", help='dataset name')

    parser.add_argument('--path', type=str, metavar='PATH', default="./datasets/SMD/processed", help='path where the dataset is located')

    parser.add_argument('--save_path', type=str, metavar='PATH', default="./checkpoints", help='path where the estimator is/should be saved')

    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', default="default_hyperparameters.json",help='path of the file of hyperparameters to use; ' + 'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,help='activate to load the estimator instead of training it')

    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    return args

# for cuda
# python univariate.py --path ./datasets/ucr/ --dataset Worms --save_path ./checkpoints --hyper default_hyperparameters.json --cuda --gpu 0

# python univariate_smd.py --path  --dataset machine-1-1 --save_path ./checkpoints --hyper default_hyperparameters.json 

if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("gpu:{}".format(args.gpu))
    else:
        args.device = torch.device("cpu")
        print("Proceeding without cuda...")


    (train, _,), (test, test_labels) = load_SMD_dataset(args.path, args.dataset)

    if not args.load:
        encoder = fit_hyperparameters(
                args.hyper, train, args.device
            )
        encoder.save_encoder(os.path.join(args.save_path, "test"))
    else:
        encoder = scikit_wrappers.CausalCNNEncoder()
        hf = open("default_hyperparameters.json", 'r')
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        encoder.set_params(**hp_dict)
        encoder.load_encoder(os.path.join(args.save_path, "test"))

    test_windows_batcher = BatchSlidingWindow(test.shape[0],window_size=encoder.window_size, batch_size=1000, shuffle=False)

    print(test.shape)
    features = encoder.encode(test_windows_batcher, test, batch_size=1000)
    print(features.shape)
    
