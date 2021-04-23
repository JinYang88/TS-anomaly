import os
import pickle
import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import (
    get_variables_as_dict,
    register_config_arguments,
    Config,
)
from pprint import pformat, pprint
from .eval_methods import pot_eval, bf_search
from .model import OmniAnomaly
from .prediction import Predictor
from .training import Trainer
from tensorflow.python.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(
        self,
        data_array,
        batch_size=32,
        shuffle=False,
    ):
        self.darray = data_array
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_pool = list(range(self.darray.shape[0]))
        self.length = int(np.ceil(len(self.index_pool) * 1.0 / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        indexes = self.index_pool[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X = self.darray[indexes]

        # in case on_epoch_end not be called automatically :)
        if index == self.length - 1:
            self.on_epoch_end()
        return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_pool)


class OmniDetector:
    def __init__(self, config):
        self.config = config
        self.time_tracker = {}
        tf.reset_default_graph()
        with tf.variable_scope("model") as model_vs:
            model = OmniAnomaly(config=self.config, name="model")

            # construct the trainer
            self.trainer = Trainer(
                model=model,
                model_vs=model_vs,
                max_epoch=self.config.max_epoch,
                batch_size=self.config.batch_size,
                valid_batch_size=self.config.test_batch_size,
                initial_lr=self.config.initial_lr,
                lr_anneal_epochs=self.config.lr_anneal_epoch_freq,
                lr_anneal_factor=self.config.lr_anneal_factor,
                grad_clip_norm=self.config.gradient_clip_norm,
                valid_step_freq=self.config.valid_step_freq,
            )

            # construct the predictor
            self.predictor = Predictor(
                model,
                batch_size=self.config.batch_size,
                n_z=self.config.test_n_z,
                last_point_only=True,
            )

    def fit(self, iterator):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.variable_scope("model") as model_vs:
            with tf.Session(config=tf_config).as_default():
                if self.config.restore_dir is not None:
                    # Restore variables from `save_dir`.
                    saver = VariableSaver(
                        get_variables_as_dict(model_vs), self.config.restore_dir
                    )
                    saver.restore()

                best_valid_metrics = self.trainer.fit(iterator)

                self.time_tracker["train"] = best_valid_metrics["total_train_time"]
                if self.config.save_dir is not None:
                    # save the variables
                    var_dict = get_variables_as_dict(model_vs)
                    saver = VariableSaver(var_dict, self.config.save_dir)
                    saver.save()
                print("=" * 30 + "result" + "=" * 30)

    def predict_prob(self, iterator):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.variable_scope("model") as model_vs:
            with tf.Session(config=tf_config).as_default():
                if self.config.save_dir is not None:
                    # Restore variables from `save_dir`.
                    saver = VariableSaver(
                        get_variables_as_dict(model_vs), self.config.save_dir
                    )
                    saver.restore()

                score, z, pred_time = self.predictor.get_score(iterator)
                self.time_tracker["test"] = pred_time
        return score
