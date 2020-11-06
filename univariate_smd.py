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
import logging

from IPython import embed
from common import scikit_wrappers, preprocessor 
from common.utils import print_to_json
from common.dataloader import load_SMD_dataset
from common.sliding import BatchSlidingWindow, WindowIterator
from common.config import parse_arguments, set_logger, initialize_config


# python univariate_smd.py 
if __name__ == '__main__':
    args = parse_arguments()
    
    # load config
    config_dir = "./hypers/" if not args["load"] else args["load"]
    params = initialize_config(config_dir, args)

    # load & preprocess data
    data_dict = load_SMD_dataset(params["path"], params["dataset"],use_dim=0)
    data_dict = preprocessor.discretize(data_dict)
    vocab_size = preprocessor.build_vocab(data_dict)
    train_windows, test_windows = preprocessor.preprocess_SMD(data_dict, window_size=params["window_size"])
    train_iterator = WindowIterator(train_windows, batch_size=params["batch_size"], shuffle=True)
    test_iterator = WindowIterator(test_windows, batch_size=params["batch_size"], shuffle=False)
    params['in_channels'] = data_dict["dim"]

    logging.info("Proceeding using {}...".format(params["device"]))
    logging.info(print_to_json(params))
    # training
    encoder = scikit_wrappers.CausalCNNEncoder(vocab_size=vocab_size, **params)
    if params["load"]:
        encoder.load_encoder()
    else:
        # encoder.fit(train_iterator, save_memory=False, verbose=True)
        encoder.save_encoder()

    # inference
    features = encoder.encode(test_iterator.loader)
    logging.info("Final features have shape: {}".format(features.shape))
    
