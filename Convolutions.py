import sys

from Dense import Dense
from Flatten import Flatten
from Softmax import Softmax

if sys.platform == 'darwin':
    print('Setting KMP_DUPLICATE_LIB_OK')
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from scipy.signal import convolve2d
from tensorflow import keras
from utility_functions import relu, averager, extract_averager_value, np_random_normal, \
    batch_generator

from Layer import Layer, ActivationFunction

from Model import Model


class Convolution2D(Layer):

    def __init__(self, weights=None, shape=None, activation=relu, **kwargs):

        if weights is None:
            assert shape is not None, 'Both weights and shape cannot be None'
            self.filter_size, _, self.num_channels, self.num_filters = shape
            weights = np_random_normal(0, 1 / np.sqrt(
                self.filter_size * self.filter_size * self.num_channels),
                                       size=(self.filter_size,
                                             self.filter_size,
                                             self.num_channels,
                                             self.num_filters))

        super().__init__(weights, **kwargs)
        self.filter_size, _, self.num_channels, self.num_filters = self.weights.shape
        self.activation = activation

        # With all true 1423.0891828536987
        self.use_fancy_indexing_for_feed_forward = False
        self.use_fancy_indexing_for_back_prop = False
        self.use_fancy_indexing_for_weight_update = False

    def feed_forward(self, X_batch, **kwargs):
        if self.first_feed_forward:  # First run
            self.first_feed_forward = False
            self.batch_size = len(X_batch)
            self.image_size = X_batch.shape[1]

            # The following is used as argument to out of ufuncs

            self.input_conv = np.zeros((self.batch_size, self.image_size,
                                        self.image_size,
                                        self.num_filters))

            # creating the zero padding structure once is efficient

            self.image_size_embedding_size = self.image_size + self.filter_size - 1
            self.input_zero_padded = np.zeros((self.batch_size,
                                               self.image_size_embedding_size,
                                               self.image_size_embedding_size,
                                               self.num_channels))

            z = np.arange(0, self.image_size)
            zs = np.stack([z + i for i in range(self.weights.shape[0])], 1)
            self.batch_index = np.arange(self.batch_size)[:, None, None, None, None, None, None]
            self.channel_index = np.arange(self.num_channels)[None, None, None, None, None, :, None]
            self.filter_index = np.arange(self.num_filters)[None, None, None, None, None, None, :]
            self.rows = zs[None, :, None, :, None, None, None]
            self.cols = zs[None, None, :, None, :, None, None]
            self.tmp = np.zeros(shape=(self.batch_size,
                                       self.image_size,
                                       self.image_size,
                                       self.filter_size,
                                       self.filter_size,
                                       self.num_channels,
                                       self.num_filters
                                       )
                                )

        self.input = X_batch

        # Convolution
        if self.use_fancy_indexing_for_feed_forward:
            self.input_zero_padded[:,
            self.filter_size // 2:-self.filter_size // 2 + 1,
            self.filter_size // 2:-self.filter_size // 2 + 1] \
                = self.input

            # TODO: better to loose the last index from all the fancy indices
            # and do a tensordot
            # Also compare with reshape and a np.matmul
            np.multiply(self.input_zero_padded[self.batch_index,
                                               self.rows,
                                               self.cols,
                                               self.channel_index],
                        self.weights[None, None, None, :, :, :, :],
                        out=self.tmp
                        )
            self.tmp.sum(axis=(3, 4, 5), out=self.input_conv)
        else:
            self.input_conv[:] = 0
            for batch_id in range(self.batch_size):
                for next_layer_channel_id in range(self.weights.shape[3]):
                    for current_layer_channel_id in range(self.weights.shape[2]):
                        # Sum over current layer after convolving
                        # Summation is done succesively in place on the ouput array
                        np.sum(
                            [self.input_conv[batch_id, :, :, next_layer_channel_id],
                             convolve2d(
                                 self.input[batch_id, :, :,
                                 current_layer_channel_id],
                                 self.weights[::-1, ::-1,
                                 current_layer_channel_id,
                                 next_layer_channel_id
                                 ],
                                 'same'
                                 )
                             ],
                            axis=0,
                            out=self.input_conv[batch_id, :, :,
                                next_layer_channel_id]
                            )

        # self.output = self.activation(self.input_conv)
        # self.output_d = self.activation(self.input_conv, der=True)
        self.output = self.input_conv
        return self.output

    def back_prop(self, loss_d_output):
        if self.first_back_prop:
            self.first_back_prop = False

            # The following three are used with out parameter of ufuncs
            self.loss_d_output_times_output_d = np.zeros_like(
                self.output)
            self.loss_derivative_input = np.zeros_like(self.input)
            self.loss_derivative_input2 = np.zeros_like(self.input)
            self.loss_derivative_weights = np.zeros_like(self.weights)
            self.loss_d_output_times_output_d_zero_padded = np.zeros((self.batch_size,
                                                                      self.image_size_embedding_size,
                                                                      self.image_size_embedding_size,
                                                                      self.num_filters))

        # np.multiply(loss_d_output, self.output_d,
        #            out=self.loss_d_output_times_output_d)
        self.loss_d_output_times_output_d = loss_d_output

        # correction for weights
        if self.trainable:

            self.input_zero_padded[:,
            self.filter_size // 2:-self.filter_size // 2 + 1,
            self.filter_size // 2:-self.filter_size // 2 + 1] \
                = self.input

            if self.use_fancy_indexing_for_weight_update:
                (self.input_zero_padded[
                     self.batch_index, self.rows, self.cols, self.channel_index] *
                 self.loss_d_output_times_output_d[:, :, :, None, None, None, :]) \
                    .sum(axis=(0, 1, 2), out=self.loss_derivative_weights)
            else:
                for alpha in range(self.weights.shape[0]):
                    for beta in range(self.weights.shape[1]):
                        x = self.loss_d_output_times_output_d[:, :, :, None, :] \
                            * self.input_zero_padded[:,
                              alpha:self.input_zero_padded.shape[1] - (
                                          self.filter_size - 1 - alpha),
                              beta:self.input_zero_padded.shape[2] - (self.filter_size - 1 - beta),
                              :,
                              None]
                        np.sum(x, axis=(0, 1, 2), out=self.loss_derivative_weights[alpha, beta])

        if not self.first_layer:
            self.loss_d_output_times_output_d_zero_padded[:,
            self.filter_size // 2:-self.filter_size // 2 + 1,
            self.filter_size // 2:-self.filter_size // 2 + 1] = \
                self.loss_d_output_times_output_d

            if self.use_fancy_indexing_for_back_prop:
                np.multiply(self.loss_d_output_times_output_d_zero_padded[self.batch_index,
                                                                          self.rows,
                                                                          self.cols,
                                                                          self.filter_index],
                            self.weights[None, None, None, ::-1, ::-1, :, :],
                            out=self.tmp
                            )
                self.tmp.sum(axis=(3, 4, 6), out=self.loss_derivative_input)
            else:
                self.loss_derivative_input[:] = 0
                if not self.first_layer:
                    for batch_id in range(loss_d_output.shape[0]):
                        for prev_layer_channel_id in range(self.weights.shape[2]):
                            for channel_id in range(self.weights.shape[3]):
                                np.sum(
                                    [self.loss_derivative_input[batch_id, :, :,
                                     prev_layer_channel_id],
                                     convolve2d(
                                         self.loss_d_output_times_output_d[batch_id, :, :,
                                         channel_id],
                                         self.weights[:, :, prev_layer_channel_id, channel_id],
                                         'same'
                                         )
                                     ],
                                    axis=0,
                                    out=self.loss_derivative_input[batch_id, :, :,
                                        prev_layer_channel_id]
                                    )

        return self.loss_derivative_input


if __name__ == '__main__':
    DATASET = 'CIFAR10'

    if DATASET == 'CIFAR10':
        print('Using CIFAR10')
        (X_train_full, y_train_full), (
            X_test, y_test) = keras.datasets.cifar10.load_data()
    elif DATASET == 'CIFAR100':
        print('Using CIFAR100')
        (X_train_full, y_train_full), (
            X_test, y_test) = keras.datasets.cifar100.load_data()
    else:
        print('Using Fashion_MNIST')
        (X_train_full, y_train_full), (
            X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + 1e-7
    X_train = (X_train - X_mean) / X_std
    X_valid = (X_valid - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    if len(X_train_full.shape) == 3:
        X_train = X_train[..., np.newaxis]
        X_valid = X_valid[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    y_train = y_train.flatten()
    y_valid = y_valid.flatten()
    y_test = y_test.flatten()

    input_num_channels = X_train.shape[3]
    input_image_size = X_train.shape[1]
    num_categories = len(set(list(y_train)))

    batch_size = 32
    num_steps = len(y_train) // batch_size

    import time

    for num_filters in (5,):
        np.random.seed(42)
        t0 = time.time()
        print('Training with num_filters ', num_filters)
        W1 = np_random_normal(0, 1 / np.sqrt(3 * 3 * input_num_channels),
                              size=(3, 3, input_num_channels, num_filters))
        W11 = np_random_normal(0, 1 / np.sqrt(3 * 3 * num_filters),
                               size=(3, 3, num_filters, num_filters))
        W2 = np_random_normal(0, 1 / np.sqrt(
            num_filters * input_image_size * input_image_size),
                              size=(num_filters * input_image_size * input_image_size,
                                    num_categories))
        m = Model()
        learning_rate = 0.001
        m.layers = [
            Convolution2D(weights=W1, name='Conv1', trainable=True,
                          activation=relu,
                          learning_rate=learning_rate,
                          first_layer=True
                          ),
            ActivationFunction(relu),
            Flatten(),
            Dense(output_dimension=10),
            Softmax()
            ]

        for epoch in range(1):
            t_start_epoch = time.time()
            # Train
            train_loss = averager()
            train_accuracy = averager()
            for i, (X_batch, y_batch) in enumerate(
                    batch_generator(X_train, y_train, batch_size, num_steps)):
                time_step = time.time()
                if (i + 1) % 10 == 0:
                    delta_time = time_step - t_start_epoch
                    eta = (num_steps / (i + 1) - 1) * delta_time
                    sys.stdout.write(
                        'Epoch: {} Step {}/{} Time Spent {:.2f}s Estimated Time {:.2f}s\r'.format(
                            epoch + 1, i + 1,
                            num_steps, delta_time, eta))
                loss, accuracy = m.feed_forward_and_back_prop(X_batch, y_batch)
                train_loss.send(loss)
                train_accuracy.send(accuracy)
            # Validate
            loss_averager_valid = averager()
            accuracy_averager_valid = averager()
            for X_valid_batch, y_valid_batch in batch_generator(X_valid, y_valid,
                                                                batch_size,
                                                                len(
                                                                    y_valid) //
                                                                batch_size):
                m.feed_forward(X_valid_batch, )
                loss, accuracy = m.loss(y_valid_batch)
                loss_averager_valid.send(loss.mean())
                accuracy_averager_valid.send(accuracy.mean())

            # report
            train_loss, train_accuracy, valid_loss, valid_accuracy = map(
                extract_averager_value, [
                    train_loss,
                    train_accuracy,
                    loss_averager_valid,
                    accuracy_averager_valid]
                )
            msg = 'Epoch {}: train loss {:.2f}, train acc {:.2f}, valid loss {' \
                  ':.2f}, valid acc {:.2f}, time taken {:.2f}s'.format(
                epoch + 1,
                train_loss,
                train_accuracy,
                valid_loss,
                valid_accuracy,
                time.time() - t_start_epoch
                )
            print(msg)
        t1 = time.time()
        print('Total time', t1 - t0)
