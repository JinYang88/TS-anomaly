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


import json
import logging
import math
import os

import nni
import numpy
import pandas
import torch
from IPython import embed

from common import data_preprocess
from common.config import initialize_config, parse_arguments, set_logger
from common.dataloader import load_CSV_dataset, load_SMAP_MSL_dataset, load_SMD_dataset
from common.sliding import BatchSlidingWindow, WindowIterator
from common.utils import print_to_json, update_from_nni_params
from networks.mlstm import MultiLSTMEncoder

# python univariate_smd.py
if __name__ == "__main__":
    args = parse_arguments()

    # load config
    config_dir = "./hypers/" if not args["load"] else args["load"]
    params = initialize_config(config_dir, args)
    params = update_from_nni_params(params, nni.get_next_parameter())

    # load & preprocess data
    data_dict = load_SMD_dataset(
        params["path"], params["dataset"], params.get("use_dim", "all")
    )
    pp = data_preprocess.preprocessor()
    if params["discretized"]:
        data_dict = pp.discretize(data_dict, params.get("n_bins", None))
        vocab_size = pp.build_vocab(data_dict)
        params["vocab_size"] = vocab_size
    if params["normalize"]:
        data_dict = pp.normalize(data_dict, method=params["normalize"])
    pp.save(params["save_path"])

    window_dict = data_preprocess.generate_windows(
        data_dict, data_hdf5_path=params["path"], **params
    )

    train_iterator = WindowIterator(
        window_dict["train_windows"], batch_size=params["batch_size"], shuffle=True
    )
    test_iterator = WindowIterator(
        window_dict["test_windows"], batch_size=params["batch_size"], shuffle=False
    )
    params["in_channels"] = data_dict["dim"]

    logging.info("Proceeding using {}...".format(params["device"]))
    logging.info(print_to_json(params))
    # training
    encoder = MultiLSTMEncoder(**params)

    if params["load"]:
        encoder.load_encoder()
    else:
        encoder.fit(
            train_iterator,
            test_iterator=test_iterator.loader,
            test_labels=window_dict["test_labels"],
            **params
        )
        encoder.save_encoder()

    encoder.load_encoder()
    eval_result_dict = encoder.score(test_iterator.loader, window_dict["test_labels"])

    logfile = "./exp_results/{}.log".format(params["dataset"])
    log = "{} | AUC-{:.3f} F1-{:.3f}\n".format(
        params["trial_id"], eval_result_dict["AUC"], eval_result_dict["F1"]
    )

    with open(logfile, "a+") as fw:
        fw.write(log)

    nni.report_final_result(eval_result_dict["AUC"])
    # inference
    # features = encoder.encode(test_iterator.loader)
    # logging.info("Final features have shape: {}".format(features.shape))
