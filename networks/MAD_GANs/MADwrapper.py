import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
os.chdir("../")
sys.path.append("./")

from common.data_preprocess import generate_windows, preprocessor, generate_windows_with_index
from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint

os.chdir("./networks/MAD_GANs/")


from AD import myADclass
import utils
import json
import numpy as np
from time import time
import DR_discriminator
import model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import plotting 



def load_settings_from_file(settings):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './experiments/settings/' + settings['settings_file'] + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'))
    # check for settings missing in file
    for key in settings.keys():
        if not key in settings_loaded:
            print(key, 'not found in loaded settings - adopting value from command line defaults: ', settings[key])
            # overwrite parsed/default settings with those read from file, allowing for
    # (potentially new) default settings not present in file
    settings.update(settings_loaded)
    return settings

def get_settings(pattern,window_size,stride,datasets):
    settings = {}
    # model architecture configuration
    settings["eval_an"] = False
    settings["eval_single"] = False
    settings["seq_length"] = window_size
    if pattern == 'train':
        settings["seq_step"] = stride
    elif pattern == 'test':
        settings["seq_step"] = 1

    settings["num_signals"] = 38
    settings["normalise"] = False
    settings["scale"] = 0.1
    settings["freq_low"] = 1.0
    settings["freq_high"] = 5.0
    settings["amplitude_low"] = 0.1
    settings["amplitude_high"] = 0.9
    settings["multivariate_mnist"] = False
    settings["full_mnist"] = False
    settings["resample_rate_in_min"] = 15
    settings["hidden_units_g"] = 100
    settings["hidden_units_d"] = 100
    settings["hidden_units_e"] = 100
    settings["kappa"] = 1
    settings["latent_dim"] = 15
    settings["weight"] = 0.5
    settings["degree"] = 1
    settings["batch_mean"] = False
    settings["learn_scale"] = False
    settings["learning_rate"] = 0.05
    settings["batch_size"] = 64
    settings["num_epochs"] = 100
    settings["D_rounds"] = 1
    settings["G_rounds"] = 3
    settings["E_rounds"] = 1
    settings["shuffle"] = True
    settings["eval_mul"] = False
    settings["wrong_labels"] = False
    settings["identifier"] = datasets
    settings["sub_id"] = datasets
    settings["dp"] = False
    settings["l2norm_bound"] = 1e-05
    settings["batches_per_lot"] = 1
    settings["dp_sigma"] = 1e-05
    settings["use_time"] = False
    settings["num_generated_features"] = 38
    return settings

def fit(samples,labels,settings):
    identifier =  settings['identifier']
    batch_size = settings['batch_size']
    seq_length = settings['seq_length']
    latent_dim = settings['latent_dim']
    learning_rate = settings['learning_rate']
    l2norm_bound = settings['l2norm_bound']
    batches_per_lot = settings['batches_per_lot']
    dp_sigma = settings['dp_sigma']
    dp = settings['dp']
    num_epochs = settings['num_epochs']
    sub_id = settings['sub_id']

    print('samples_size:',samples.shape)
    # -- number of variables -- #
    num_variables = samples.shape[2]
    print('num_variables:', num_variables)
    # --- save settings, data --- #
    print('Ready to run with settings:')
    for (k, v) in settings.items(): print(v, '\t', k)
    # add the settings to local environment
    # WARNING: at this point a lot of variables appear
    locals().update(settings)
    json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0) 

    # --- build model --- #
    # preparation: data placeholders and model parameters
    Z, X, T = model.create_placeholders(batch_size, seq_length, latent_dim, num_variables)
    discriminator_vars = ['hidden_units_d', 'seq_length', 'batch_size', 'batch_mean']
    discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
    generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'learn_scale']
    generator_settings = dict((k, settings[k]) for k in generator_vars)
    generator_settings['num_signals'] = num_variables

    # model: GAN losses
    D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings)
    D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size,
                                                            total_examples=samples.shape[0],
                                                            l2norm_bound=l2norm_bound,
                                                            batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
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
    plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=16, num_epochs=num_epochs)
    plotting.save_samples_real(vis_real, identifier)

    # --- train --- #
    train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 'latent_dim']
    train_settings = dict((k, settings[k]) for k in train_vars)
    train_settings['num_signals'] = num_variables

    MMD = np.zeros([num_epochs, ])

    for epoch in range(num_epochs):
        D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples, labels, sess, Z, X, D_loss, G_loss,
                                                    D_solver, G_solver, **train_settings)    
        print('epoch, D_loss_curr, G_loss_curr, seq_length')
        print('%d\t%.4f\t%.4f\t%d' % (epoch, D_loss_curr, G_loss_curr, seq_length))
        print(sub_id)
        model.dump_parameters(sub_id + '_' + str(seq_length) + '_' + str(epoch), sess)

    np.save('./experiments/plots/gs/' + identifier + '_' + 'MMD.npy', MMD)

    print('Training terminated | Training time=%d s' %(end) )
    print("Training terminated | training time = %ds  " % (time() - begin))


def detect(samples,labels,index,settings):

    identifier =  settings['identifier']

    print('Ready to run with settings:')
    for (k, v) in settings.items(): print(v, '\t', k)
    # add the settings to local environment
    # WARNING: at this point a lot of variables appear
    locals().update(settings)
    json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

    epoch  = settings['num_epochs']-1
    ob = myADclass(epoch=epoch,samples=samples,labels=labels,index=index,settings=settings)
    anomaly_score,anomaly_label = ob.ADfunc()

    return anomaly_score,anomaly_label


