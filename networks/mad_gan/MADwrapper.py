import os
import sys
import json

sys.path.append("../")

import numpy as np
from time import time
import tensorflow.compat.v1 as tf
from networks.mad_gan.AD import myADclass
from networks.mad_gan import utils
from networks.mad_gan import DR_discriminator
from networks.mad_gan import model
from networks.mad_gan import plotting

tf.disable_v2_behavior()


class MAD_GAN:
    def __init__(self, save_dir):
        tf.reset_default_graph()
        self.save_dir = save_dir
        self.time_tracker = {}
        os.makedirs(save_dir, exist_ok=True)

    def load_settings_from_file(self, settings):
        """
        Handle loading settings from a JSON file, filling in missing settings from
        the command line defaults, but otherwise overwriting them.
        """
        settings_path = os.path.join(self.save_dir, settings["settings_file"] + ".txt")

        print("Loading settings from", settings_path)
        settings_loaded = json.load(open(settings_path, "r"))
        # check for settings missing in file
        for key in settings.keys():
            if not key in settings_loaded:
                print(
                    key,
                    "not found in loaded settings - adopting value from command line defaults: ",
                    settings[key],
                )
                # overwrite parsed/default settings with those read from file, allowing for
        # (potentially new) default settings not present in file
        settings.update(settings_loaded)
        return settings

    def fit(self, samples, labels, settings):
        identifier = settings["identifier"]
        batch_size = settings["batch_size"]
        seq_length = settings["seq_length"]
        latent_dim = settings["latent_dim"]
        learning_rate = settings["learning_rate"]
        l2norm_bound = settings["l2norm_bound"]
        batches_per_lot = settings["batches_per_lot"]
        dp_sigma = settings["dp_sigma"]
        dp = settings["dp"]
        num_epochs = settings["num_epochs"]
        sub_id = settings["sub_id"]

        print("samples_size:", samples.shape)
        # -- number of variables -- #
        num_variables = samples.shape[2]
        print("num_variables:", num_variables)
        # --- save settings, data --- #
        print("Ready to run with settings:")
        for (k, v) in settings.items():
            print(v, "\t", k)
        # add the settings to local environment
        # WARNING: at this point a lot of variables appear
        locals().update(settings)
        json.dump(
            settings,
            open(os.path.join(self.save_dir, identifier + ".txt"), "w"),
            indent=0,
        )

        # --- build model --- #
        # preparation: data placeholders and model parameters
        Z, X, T = model.create_placeholders(
            batch_size, seq_length, latent_dim, num_variables
        )
        discriminator_vars = [
            "hidden_units_d",
            "seq_length",
            "batch_size",
            "batch_mean",
        ]
        discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
        generator_vars = ["hidden_units_g", "seq_length", "batch_size", "learn_scale"]
        generator_settings = dict((k, settings[k]) for k in generator_vars)
        generator_settings["num_signals"] = num_variables

        # model: GAN losses
        D_loss, G_loss = model.GAN_loss(
            Z, X, generator_settings, discriminator_settings
        )
        D_solver, G_solver, priv_accountant = model.GAN_solvers(
            D_loss,
            G_loss,
            learning_rate,
            batch_size,
            total_examples=samples.shape[0],
            l2norm_bound=l2norm_bound,
            batches_per_lot=batches_per_lot,
            sigma=dp_sigma,
            dp=dp,
        )
        # model: generate samples for visualization
        G_sample = model.generator(Z, **generator_settings, reuse=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # # -- plot the real samples -- #
        vis_real_indices = np.random.choice(len(samples), size=16)
        vis_real = np.float32(samples[vis_real_indices, :, :])
        # plotting.save_plot_sample(
        #     vis_real, 0, identifier + "_real", n_samples=16, num_epochs=num_epochs
        # )
        # plotting.save_samples_real(vis_real, identifier)

        # --- train --- #
        train_vars = [
            "batch_size",
            "D_rounds",
            "G_rounds",
            "use_time",
            "seq_length",
            "latent_dim",
        ]
        train_settings = dict((k, settings[k]) for k in train_vars)
        train_settings["num_signals"] = num_variables

        MMD = np.zeros(
            [
                num_epochs,
            ]
        )

        begin = time()
        for epoch in range(num_epochs):
            D_loss_curr, G_loss_curr = model.train_epoch(
                epoch,
                samples,
                labels,
                sess,
                Z,
                X,
                D_loss,
                G_loss,
                D_solver,
                G_solver,
                **train_settings
            )
            print("epoch, D_loss_curr, G_loss_curr, seq_length")
            print("%d\t%.4f\t%.4f\t%d" % (epoch, D_loss_curr, G_loss_curr, seq_length))
            print(sub_id)
            model.dump_parameters(sub_id + "_" + str(seq_length), sess)
        self.time_tracker["train"] = time() - begin
        print("Training terminated | training time = %ds  " % (time() - begin))

    def detect(self, samples, labels, index, settings):

        identifier = settings["identifier"]

        print("Ready to run with settings:")
        for (k, v) in settings.items():
            print(v, "\t", k)
        # add the settings to local environment
        # WARNING: at this point a lot of variables appear
        locals().update(settings)
        json.dump(
            settings,
            open(os.path.join(self.save_dir, identifier + ".txt"), "w"),
            indent=0,
        )

        epoch = settings["num_epochs"] - 1
        ob = myADclass(
            epoch=epoch, samples=samples, labels=labels, index=index, settings=settings
        )

        begin = time()
        anomaly_score, anomaly_label = ob.ADfunc()
        self.time_tracker["test"] = time() - begin

        return anomaly_score, anomaly_label
