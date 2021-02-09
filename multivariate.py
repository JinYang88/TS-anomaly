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
import pandas as pd
from IPython import embed

from common import data_preprocess
from common.config import (
    initialize_config,
    parse_arguments,
    set_logger,
    subdatasets,
    get_trial_id,
)
from common.dataloader import load_CSV_dataset, load_dataset
from common.sliding import BatchSlidingWindow, WindowIterator
from common.utils import print_to_json, update_from_nni_params, seed_everything
from networks.mlstm import MultiLSTMEncoder

seed_everything(2020)


def run(params):
    # load & preprocess data
    data_dict = load_dataset(
        params["dataset"],
        params["subdataset"],
        params.get("use_dim", "all"),
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
    records = encoder.score(test_iterator.loader, window_dict["test_labels"])

    records.update(
        {
            "trial_id": params["trial_id"],
            "expid": params["expid"],
            "dataset": "{}-{}".format(params["dataset"], params["subdataset"]),
            "set": params["set"],
        }
    )

    logfile = "./experiment_results.csv"
    log = "{}\t{}\t{}\t{}\tAUC-{:.3f}\tF1-{:.3f}\tF1adj-{:.3f}\n".format(
        records["trial_id"],
        records["set"],
        records["expid"],
        records["dataset"],
        records["AUC"],
        records["F1"],
        records["F1_adj"],
    )

    with open(logfile, "a+") as fw:
        fw.write(log)

    nni.report_final_result(records["AUC"])

    # inference
    # features = encoder.encode(test_iterator.loader)
    # logging.info("Final features have shape: {}".format(features.shape))
    return records


# python univariate_smd.py
if __name__ == "__main__":
    args = parse_arguments()
    # load config
    config_dir = "./hypers/" if not args["load"] else args["load"]
    params = initialize_config(config_dir, args)
    params = update_from_nni_params(params, nni.get_next_parameter())

    if not params["subdataset"]:
        detail_dir = "./details"
        os.makedirs(detail_dir, exist_ok=True)
        start_time = get_trial_id()
        records = []
        # run each subdataset
        for dataset in subdatasets[params["dataset"]]:
            params["subdataset"] = dataset
            records.append(run(params))
        records = pd.DataFrame(records)
        records.to_csv(
            "./{}/{}-{}-all.csv".format(detail_dir, params["dataset"], start_time),
            index=False,
        )

        log = "{}\t{}\t{}\t{}\tAUC-{:.3f}\tF1-{:.3f}\tF1_adj-{:.3f}\tPS_adj-{:.3f}\tRC_adj-{:.3f}\n".format(
            start_time,
            params["set"],
            params["expid"],
            params["dataset"] + "_all",
            records["AUC"].mean(),
            records["F1"].mean(),
            records["F1_adj"].mean(),
            records["PS_adj"].mean(),
            records["RC_adj"].mean(),
        )
        with open("./total_results.csv", "a+") as fw:
            fw.write(log)

    else:
        run(params)