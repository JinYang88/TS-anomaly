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
from common.dataloader import load_SMD_dataset
from common import scikit_wrappers
from common.sliding import BatchSlidingWindow, WindowIterator
from common.preprocessor import preprocess_SMD
from common.config import parse_arguments, set_logger



    # preprocess
    # output: train_windows test_windows

    return 


# python univariate_smd.py 
if __name__ == '__main__':
    args = parse_arguments()
    set_logger(args)
    logging.info("Proceeding using {}...".format(args.device))

    # load config
    with open(os.path.join(args.hyper), 'r') as hf
        params = json.load(hf)
    
    # load & preprocess data
    data_dict = load_SMD_dataset(args.path, args.dataset, use_dim=0)
    train_windows, test_windows = preprocess_SMD(data_dict, window_size=params["window_size"])
    train_iterator = WindowIterator(train_windows, batch_size=params["batch_size"], shuffle=True)
    test_iterator = WindowIterator(test_windows, batch_size=params["batch_size"], shuffle=False)
    params['in_channels'] = data_dict["dim"]

    # training
    encoder = scikit_wrappers.CausalCNNEncoder()
    encoder.set_params(**params)
    encoder.fit(
        train_iterator, save_memory=save_memory, verbose=True
    )


    features = encoder.encode(test_iterator, batch_size=1000)
    logging.info("final features have shape: {}".format(features.shape))
    
