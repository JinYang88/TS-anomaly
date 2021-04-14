import os
import time
import argparse
import logging
import glob
import yaml

subdatasets = {
    "SMD": ["machine-1-{}".format(i) for i in range(1, 9)]
    + ["machine-2-{}".format(i) for i in range(1, 10)]
    + ["machine-3-{}".format(i) for i in range(1, 12)],
    "SMAP": [
        "P-1",
        "S-1",
        "E-1",
        "E-2",
        "E-3",
        "E-4",
        "E-5",
        "E-6",
        "E-7",
        "E-8",
        "E-9",
        "E-10",
        "E-11",
        "E-12",
        "E-13",
        "A-1",
        "D-1",
        "P-2",
        "P-3",
        "D-2",
        "D-3",
        "D-4",
        "A-2",
        "A-3",
        "A-4",
        "G-1",
        "G-2",
        "D-5",
        "D-6",
        "D-7",
        "F-1",
        "P-4",
        "G-3",
        "T-1",
        "T-2",
        "D-8",
        "D-9",
        "F-2",
        "G-4",
        "T-3",
        "D-11",
        "D-12",
        "B-1",
        "G-6",
        "G-7",
        "P-7",
        "R-1",
        "A-5",
        "A-6",
        "A-7",
        "D-13",
        "P-2",
        "A-8",
        "A-9",
        "F-3",
    ],
    "MSL": [
        "M-6",
        "M-1",
        "M-2",
        "S-2",
        "P-10",
        "T-4",
        "T-5",
        "F-7",
        "M-3",
        "M-4",
        "M-5",
        "P-15",
        "C-1",
        "C-2",
        "T-12",
        "T-13",
        "F-4",
        "F-5",
        "D-14",
        "T-9",
        "P-14",
        "T-8",
        "P-11",
        "D-15",
        "D-16",
        "M-7",
        "F-8",
    ],
    "WADI": ["wadi"],
    "SWAT": ["swat"],
}


def parse_multi_setting(setting):
    result_dict = {}
    if setting:
        for item in setting:
            k, v = item.strip().split("=")
            try:
                result_dict[k] = eval(v)
            except NameError:
                result_dict[k] = str(v)
    return result_dict


def parse_arguments():
    import torch

    parser = argparse.ArgumentParser(
        description="Anomaly detection repository for TS datasets"
    )
    parser.add_argument(
        "--subdataset",
        type=str,
        metavar="D",
        default="",
        help="dataset name",
    )

    parser.add_argument(
        "--dataset", type=str, metavar="D", default="SMD", help="dataset name"
    )

    # SMD: "./datasets/anomaly/SMD/processed"
    # SMAP: "./datasets/anomaly/SMAP-MSL/processed_SMAP"
    # MSL: "./datasets/anomaly/SMAP-MSL/processed_MSL"
    # Simulated: "./datasets/Simulated/simulated_p0.1.csv"
    parser.add_argument(
        "--path",
        type=str,
        metavar="PATH",
        # default="./datasets/anomaly/SMAP-MSL/",
        default="./datasets/anomaly/SMAP-MSL/processed_SMAP",
        help="path where the dataset is located",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        metavar="PATH",
        default="./checkpoints",
        help="path where the estimator is/should be saved",
    )

    parser.add_argument("--set", metavar="KEY=VALUE", nargs="+")

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="GPU",
        help="index of GPU used for computations (default: 0)",
    )
    parser.add_argument(
        "--expid", type=str, default="mlstm_250", help="Expid in hypers"
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Load a pretrained model from the specific directory",
    )
    parser.add_argument(
        "--nrows", type=int, default=None, help="Read only first nrows for test"
    )
    parser.add_argument(
        "--clear",
        type=int,
        default=None,
        help="Set to 1 if data re-processsing is needed",
    )

    args = parser.parse_args()
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        args.device = torch.device("cpu")
    os.makedirs(args.save_path, exist_ok=True)
    return vars(args)


def initialize_config(config_dir, args):
    params = dict()
    model_configs = glob.glob(os.path.join(config_dir, "*/*.yaml")) + glob.glob(
        os.path.join(config_dir, "*.yaml")
    )

    if not model_configs:
        raise RuntimeError("config_dir={} is not valid!".format(config_dir))
    found_params = find_config(model_configs, args["expid"])
    base_config = found_params.get("Base", {})
    model_config = found_params.get(args["expid"])
    params.update(base_config)
    params.update(args)
    params.update(model_config)

    params = set_logger(params)
    params.update(parse_multi_setting(args["set"]))

    with open(os.path.join(params["save_path"], "model_config.yaml"), "w") as fr:
        found_params["Base"]["save_path"] = params["save_path"]
        found_params["Base"]["trial_id"] = params["trial_id"]
        yaml.dump(found_params, fr)

    if params["dataset"] in ["SMAP", "MSL"]:
        params["prediction_dims"] = [0]
    else:
        params["prediction_dims"] = []

    return params


def get_trial_id():
    trial_id = nni.get_trial_id()
    if trial_id == "STANDALONE":
        trial_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    return trial_id


def set_logger(params):
    if not params["load"]:
        trial_id = get_trial_id()
        log_dir = os.path.join(params["save_path"], trial_id)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = params["save_path"]
        trial_id = params["trial_id"]
    log_file = os.path.join(log_dir, "{}.log".format(trial_id))

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    # update save_path
    params["save_path"] = log_dir
    params["trial_id"] = trial_id
    return params


def find_config(model_configs, experiment_id):
    found_params = dict()
    for config in model_configs:
        with open(config, "r", encoding="utf-8") as cfg:
            config_dict = yaml.safe_load(cfg)
            if "Base" in config_dict:
                found_params["Base"] = config_dict["Base"]
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    return found_params