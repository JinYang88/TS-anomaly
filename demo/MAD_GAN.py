import os
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
sys.path.append("./")

from common.data_preprocess import generate_windows, preprocessor, generate_windows_with_index
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint

dataset = "SMD"
subdataset = "machine-1-1"

data_dict = load_dataset(
            dataset,
            subdataset,
            "all",
        )

os.chdir("./networks/MAD_GANs/")

from AD import myADclass
from MADwrapper import get_settings,fit,detect
import utils
import json
import numpy as np
from time import time
import DR_discriminator
import model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import plotting 

begin = time()



if __name__ == "__main__":
    window_size = 32
    stride = 5

    pp = preprocessor()
    data_dict = pp.normalize(data_dict)

    # generate sliding windows
    window_dict = generate_windows_with_index(data_dict, window_size=window_size, stride=stride)

    train = window_dict["train_windows"]
    test = window_dict["test_windows"]
    test_labels = window_dict["test_labels"]
    index = window_dict["index_windows"]
    labels = np.zeros([train.shape[0],train.shape[1], 1])

    settings = get_settings('train',window_size,stride,"smd_train_machine_1_1")
    samples = train
    fit(samples,labels,settings)

    settings = get_settings('test',window_size,stride,"smd_train_machine_1_1")
    samples = test
    labels = test_labels
    labels = labels.reshape([labels.shape[0],labels.shape[1],1])
    index = index.reshape([index.shape[0],index.shape[1],1])

    anomaly_score,anomaly_label = detect(samples,labels,index,settings)
    
    eva = evaluator(
        ["auc", "f1", "pc", "rc"],
        anomaly_score,
        anomaly_label,
        iterate_threshold=True,
        iterate_metric="f1",
        point_adjustment=True,
    )

    eval_results = eva.compute_metrics()
    
    pprint(eval_results)