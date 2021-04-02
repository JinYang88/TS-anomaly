import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pdb
import json
import model
import os
import sys

import utils
import eval
import DR_discriminator
import data_utils

# from pyod.utils.utility import *
from sklearn.utils.validation import *
from sklearn.metrics.classification import *
from sklearn.metrics.ranking import *
from time import time

from common.evaluation import evaluator


begin = time()

"""
Here, only the discriminator was used to do the anomaly detection
"""

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser('')
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults


class myADclass():
    def __init__ (self, epoch,samples, labels, index,settings=settings):
        self.epoch = epoch
        self.settings = settings
        self.samples = samples
        self.labels = labels
        self.index = index
    def ADfunc(self):
        num_samples_t = self.samples.shape[0]
        print('sample_shape:', self.samples.shape[0])
        print('num_samples_t', num_samples_t)

        # -- only discriminate one batch for one time -- #
        D_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        DL_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        L_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        I_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        batch_times = num_samples_t // self.settings['batch_size']
        for batch_idx in range(0, num_samples_t // self.settings['batch_size']):
            # print('batch_idx:{}
            # display batch progress
            model.display_batch_progression(batch_idx, batch_times)
            start_pos = batch_idx * self.settings['batch_size']
            end_pos = start_pos + self.settings['batch_size']
            T_mb = self.samples[start_pos:end_pos, :, :]
            L_mmb = self.labels[start_pos:end_pos, :, :]
            I_mmb = self.index[start_pos:end_pos, :, :]
            para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
                self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
            D_t, L_t = DR_discriminator.dis_trained_model(self.settings, T_mb, para_path)
            D_test[start_pos:end_pos, :, :] = D_t
            DL_test[start_pos:end_pos, :, :] = L_t
            L_mb[start_pos:end_pos, :, :] = L_mmb
            I_mb[start_pos:end_pos, :, :] = I_mmb

        start_pos = (num_samples_t // self.settings['batch_size']) * self.settings['batch_size']
        end_pos = start_pos + self.settings['batch_size']
        size = self.samples[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([self.settings['batch_size'] - size, self.samples.shape[1], self.samples.shape[2]])
        batch = np.concatenate([self.samples[start_pos:end_pos, :, :], fill], axis=0)
        para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
            self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
        D_t, L_t = DR_discriminator.dis_trained_model(self.settings, batch, para_path)
        L_mmb = self.labels[start_pos:end_pos, :, :]
        I_mmb = self.index[start_pos:end_pos, :, :]
        D_test[start_pos:end_pos, :, :] = D_t[:size, :, :]
        DL_test[start_pos:end_pos, :, :] = L_t[:size, :, :]
        L_mb[start_pos:end_pos, :, :] = L_mmb
        I_mb[start_pos:end_pos, :, :] = I_mmb

        

        results = np.zeros([18, 4])
        
        anomaly_score = DL_test.reshape([D_test.shape[0],D_test.shape[1]])
        anomaly_score = anomaly_score.mean(axis=-1)
        anomaly_label = L_mb.reshape([L_mb.shape[0],L_mb.shape[1]])
        anomaly_label = anomaly_label[:, -1]

        eva = evaluator(
        ["auc", "f1", "pc", "rc"],
        anomaly_score,
        anomaly_label,
        iterate_threshold=True,
        iterate_metric="f1",
        point_adjustment=True,
    )
        eval_results = eva.compute_metrics()


        return results

if __name__ == "__main__":
    print('Main Starting...')

    Results = np.empty([settings['num_epochs'], 18, 4])

    for epoch in range(settings['num_epochs']):
    # for epoch in range(50, 60):
        ob = myADclass(epoch)
        Results[epoch, :, :] = ob.ADfunc()

    res_path = './experiments/plots/Results' + '_' + settings['sub_id'] + '_' + str(
        settings['seq_length']) + '.npy'
    np.save(res_path, Results)

    print('Main Terminating...')
    end = time() - begin
    print('Testing terminated | Training time=%d s' % (end))