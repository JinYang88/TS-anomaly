import os
import time
import argparse
import logging
import torch
import nni
import glob
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Anomaly detection repository for TS datasets')

    parser.add_argument('--dataset', type=str, metavar='D', default="machine-1-1", help='dataset name')

    parser.add_argument('--path', type=str, metavar='PATH', default="./datasets/SMD/processed", help='path where the dataset is located')

    parser.add_argument('--save_path', type=str, metavar='PATH', default="./checkpoints", help='path where the estimator is/should be saved')

    parser.add_argument('--gpu', type=int, default=0, metavar='GPU', help='index of GPU used for computations (default: 0)')
    parser.add_argument('--expid', type=str, default="casualCnn",help='Expid in hypers')
    parser.add_argument('--load', action='store_true', default=False,help='activate to load the estimator instead of training it')

    args = parser.parse_args()
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        args.device = torch.device("cpu")
    os.makedirs(args.save_path, exist_ok=True)
    return args


def initialize_config(config_dir, args):
    params = dict()
    model_configs = glob.glob(os.path.join(config_dir, '*/*.yaml')) + glob.glob(os.path.join(config_dir, '*.yaml'))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = find_config(model_configs, args.expid)
    base_config = found_params.get('Base', {})
    model_config = found_params.get(args.expid) 
    params.update(base_config)
    params.update(model_config)
    params.update(vars(args))

    log_dir, trial_id = set_logger(params)

    with open(os.path.join(log_dir, "model_config.yaml"), "w") as fr:
        yaml.dump(found_params, fr) 

    # update save_path
    params["save_path"] = log_dir
    params["trial_id"] = trial_id
    return params


def set_logger(params):
    trial_id = nni.get_trial_id()
    if trial_id == "STANDALONE":
        trial_id = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    log_dir = os.path.join(params["save_path"], trial_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "{}.log".format(trial_id))

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

    return log_dir, trial_id


def find_config(model_configs, experiment_id):
    found_params = dict()
    for config in model_configs:
        with open(config, 'r', encoding="utf-8") as cfg:
            config_dict = yaml.safe_load(cfg)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    return found_params