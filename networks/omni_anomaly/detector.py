import os
import pickle
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
from .utils import get_data, save_z
from tensorflow.python.keras.utils import Sequence


class OmniDetector:
    def __init__(self, config):
        tf.reset_default_graph()
        self.config = config

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
        with tf.variable_scope("model") as model_vs:
            with tf.Session().as_default():
                if self.config.restore_dir is not None:
                    # Restore variables from `save_dir`.
                    saver = VariableSaver(
                        get_variables_as_dict(model_vs), self.config.restore_dir
                    )
                    saver.restore()

                best_valid_metrics = self.trainer.fit(iterator)

                if self.config.save_dir is not None:
                    # save the variables
                    var_dict = get_variables_as_dict(model_vs)
                    saver = VariableSaver(var_dict, self.config.save_dir)
                    saver.save()
                print("=" * 30 + "result" + "=" * 30)

    def predict_prob(self, iterator):
        with tf.variable_scope("model") as model_vs:
            with tf.Session().as_default():
                score, z, pred_speed = self.predictor.get_score(iterator)
        return score
