from __future__ import print_function
import numpy as np
import tensorflow as tf
import six
from timeit import default_timer as timer


class LSTM_Var_Autoencoder(object):
    def __init__(
        self,
        intermediate_dim=None,
        z_dim=None,
        n_dim=None,
        kulback_coef=0.1,
        stateful=False,
    ):
      
        self.z_dim = z_dim
        self.n_dim = n_dim
        self.intermediate_dim = intermediate_dim
        self.stateful = stateful
        self.input = tf.placeholder(tf.float32, shape=[None, None, self.n_dim])
        self.batch_size = tf.placeholder(tf.int64)
        self.kulback_coef = kulback_coef
        # tf.data api
        dataset = (
            tf.data.Dataset.from_tensor_slices(self.input)
            .repeat()
            .batch(self.batch_size)
        )
        self.batch_ = tf.placeholder(tf.int32, shape=[])
        self.ite = dataset.make_initializable_iterator()
        self.x = self.ite.get_next()
        self.repeat = tf.placeholder(tf.int32)

        def gauss_sampling(mean, sigma):
            with tf.name_scope("sample_gaussian"):
                eps = tf.random_normal(tf.shape(sigma), 0, 1, dtype=tf.float32)
                # It should be log(sigma / 2), but this empirically converges"
                # much better for an unknown reason"
                z = tf.add(mean, tf.exp(0.5 * sigma) * eps)
                return z

    def fit(
        self,
        X,
        learning_rate=0.001,
        batch_size=100,
        num_epochs=200,
        opt=tf.train.AdamOptimizer,
        REG_LAMBDA=0,
        grad_clip_norm=10,
        optimizer_params=None,
        verbose=True,
    ):

        if len(np.shape(X)) != 3:
            raise ValueError(
                "Input must be a 3-D array. I could reshape it for you, but I am too lazy."
                " \n            Use input.reshape(-1,timesteps,1)."
            )
        if optimizer_params is None:
            optimizer_params = {}
            optimizer_params["learning_rate"] = learning_rate
        else:
            optimizer_params = dict(six.iteritems(optimizer_params))

        self._create_loss_optimizer(opt, **optimizer_params)
        lstm_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="LSTM_encoder"
        )
        self._cost += REG_LAMBDA * tf.reduce_mean(tf.nn.l2_loss(lstm_var))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(
            self.ite.initializer, feed_dict={self.input: X, self.batch_size: batch_size}
        )
        batches_per_epoch = int(np.ceil(len(X) / batch_size))

        print("\n")
        print("Training...")
        print("\n")
        start = timer()

        for epoch in range(num_epochs):
            train_error = 0
            for step in range(batches_per_epoch):
                if self.stateful:
                    loss, _, s, _ = self.sess.run(
                        [self._cost, self.train_op, self.update_op, self.update_op_],
                        feed_dict={
                            self.repeat: np.shape(X)[1],
                            self.batch_: batch_size,
                        },
                    )
                else:
                    loss, _ = self.sess.run(
                        [self._cost, self.train_op],
                        feed_dict={self.repeat: np.shape(X)[1]},
                    )
                train_error += loss
            if step == (batches_per_epoch - 1):
                mean_loss = train_error / batches_per_epoch

                if self.stateful:  # reset cell & hidden states between epochs
                    self.sess.run(
                        [self.reset_state_op], feed_dict={self.batch_: batch_size}
                    )
                    self.sess.run(
                        [self.reset_state_op_], feed_dict={self.batch_: batch_size}
                    )
            if epoch % 10 == 0 & verbose:
                print("Epoch {:^6} Loss {:0.5f}".format(epoch + 1, mean_loss))
        end = timer()
        print("\n")
        print("Training time {:0.2f} minutes".format((end - start) / (60)))

    def predict_prob(self, X):
        self.sess.run(
            self.ite.initializer,
            feed_dict={self.input: X, self.batch_size: np.shape(X)[0]},
        )
        if self.stateful:
            _, _ = self.sess.run(
                [self.reset_state_op, self.reset_state_op_],
                feed_dict={self.batch_: np.shape(X)[0]},
            )
            x_rec, _, _ = self.sess.run(
                [self.x_reconstr_mean, self.update_op, self.update_op_],
                feed_dict={self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]},
            )
        else:
            x_rec = self.sess.run(
                self.x_reconstr_mean, feed_dict={self.repeat: np.shape(X)[1]}
            )
        squared_error = (x_rec - X) ** 2
        return squared_error

    def reconstruct(self, X, get_error=False):
        self.sess.run(
            self.ite.initializer,
            feed_dict={self.input: X, self.batch_size: np.shape(X)[0]},
        )
        if self.stateful:
            _, _ = self.sess.run(
                [self.reset_state_op, self.reset_state_op_],
                feed_dict={self.batch_: np.shape(X)[0]},
            )
            x_rec, _, _ = self.sess.run(
                [self.x_reconstr_mean, self.update_op, self.update_op_],
                feed_dict={self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]},
            )
        else:
            x_rec = self.sess.run(
                self.x_reconstr_mean, feed_dict={self.repeat: np.shape(X)[1]}
            )
        if get_error:
            squared_error = (x_rec - X) ** 2
            return x_rec, squared_error
        else:
            return x_rec

    def reduce(self, X):
        self.sess.run(
            self.ite.initializer,
            feed_dict={self.input: X, self.batch_size: np.shape(X)[0]},
        )
        if self.stateful:
            _ = self.sess.run(
                [self.reset_state_op], feed_dict={self.batch_: np.shape(X)[0]}
            )
            x, _ = self.sess.run(
                [self.z, self.update_op],
                feed_dict={self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]},
            )
        else:
            x = self.sess.run(self.z)
        return x
