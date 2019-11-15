import sys

import numpy as np
from scipy.signal import convolve2d
from tensorflow import keras
from utility_functions import relu, relu_prime, averager, extract_averager_value


def batch_generator(X, y, batch_size, total_count):
    idx = np.arange(0, len(y))
    for i in range(total_count):
        idx_batch = np.random.choice(idx, batch_size)
        yield X[idx_batch], y[idx_batch]


class Layer(object):

    def __init__(self, weights, trainable=True, learning_rate=0.001, name=None):
        self.learning_rate = learning_rate
        self.weights = weights.copy()
        self.first_feed_forward = True
        self.first_back_prop = True
        self.first_layer = False
        self.trainable = trainable
        self.name = name
        if not self.trainable:
            print('{} not trainable'.format(self))

    def __repr__(self):
        return '{}: {}'.format(self.__class__, self.name)

    def feed_forward(self, prev_layer):
        raise NotImplementedError

    def back_prop(self, next_layer_loss_gradient):
        raise NotImplementedError

    def update_weights(self):
        if self.trainable:
            np.add(self.weights,
                   -self.learning_rate * self.loss_derivative_weights,
                   out=self.weights)

    def set_first_layer(self):
        if not self.first_layer:
            self.first_layer = True
            print('{} set as first layer'.format(self))


class Convolution2D(Layer):

    def __init__(self, weights=None, shape=None, **kwargs):

        if weights is None:
            assert shape is not None, 'Both weights and shape cannot be None'
            self.filter_size, _, self.num_channels, self.num_filters = shape
            weights = np.random.normal(0, 1 / np.sqrt(
                self.filter_size * self.filter_size * self.num_channels),
                                       size=(self.filter_size,
                                             self.filter_size,
                                             self.num_channels,
                                             self.num_filters))

        super().__init__(weights, **kwargs)
        self.filter_size, _, self.num_channels, self.num_filters = self.weights.shape
        self.batch_size = None
        self.image_size_embedding_size = None

    def feed_forward(self, X_batch):
        if self.first_feed_forward:  # First run
            self.first_feed_forward = False
            self.batch_size = len(X_batch)
            self.image_size = X_batch.shape[1]
            self.input_conv = np.zeros((self.batch_size, self.image_size,
                                        self.image_size,
                                        self.num_filters))
            self.output = np.zeros_like(self.input_conv)
            self.output_d = np.zeros_like(self.input_conv)

        self.input = X_batch

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

        self.output[:] = relu(self.input_conv)
        self.output_d[:] = relu_prime(self.input_conv)
        return self.output

    def back_prop(self, loss_d_output):
        if self.first_back_prop:
            self.first_back_prop = False
            self.image_size_embedding_size = self.image_size + self.filter_size - 1
            self.input_zero_padded = np.zeros((self.batch_size,
                                               self.image_size_embedding_size,
                                               self.image_size_embedding_size,
                                               self.num_channels))
            self.loss_d_output_times_output_d = np.zeros_like(
                self.output_d)
            self.loss_derivative_input = np.zeros_like(self.input)
            self.loss_derivative_weights=np.zeros_like(self.weights)

        np.multiply(loss_d_output, self.output_d,
                    out=self.loss_d_output_times_output_d)
        self.input_zero_padded[:,
        self.filter_size // 2:-self.filter_size // 2 + 1,
        self.filter_size // 2:-self.filter_size // 2 + 1] \
            = self.input

        if self.trainable:

            for alpha in range(self.weights.shape[0]):
                for beta in range(self.weights.shape[1]):
                    x = self.loss_d_output_times_output_d[:, :, :, None,
                        :] * self.input_zero_padded[:, alpha:self.input_zero_padded.shape[1] - (
                                self.filter_size - 1 - alpha), beta:self.input_zero_padded.shape[2] - (self.filter_size - 1 -
                                                                   beta),
                                                    :, None]
                    np.sum(x, axis=(0, 1, 2), out=self.loss_derivative_weights[alpha, beta])

        self.loss_derivative_input[:] = 0
        if not self.first_layer:
            for batch_id in range(loss_d_output.shape[0]):
                for prev_layer_channel_id in range(self.weights.shape[2]):
                    for channel_id in range(self.weights.shape[3]):
                        np.sum(
                            [self.loss_derivative_input[batch_id, :, :,prev_layer_channel_id],
                            convolve2d(
                                self.loss_d_output_times_output_d[batch_id, :, :, channel_id],
                                self.weights[:, :, prev_layer_channel_id,channel_id],
                                'same'
                            )
                            ],
                            axis=0,
                            out=self.loss_derivative_input[batch_id, :, :,prev_layer_channel_id]
                        )
        return self.loss_derivative_input


class DenseSoftmax(Layer):

    def __init__(self, weights=None, shape=None,
                 assume_cross_entropy_loss=True, **kwargs):

        if weights is None:
            assert shape is not None, 'Both weights and shape cannot be None'
            weights = np.random.normal(0, 1, size=shape)

        super().__init__(weights, **kwargs)
        self.num_categories = self.weights.shape[1]
        self.batch_size = None
        self.assume_cross_entropy_loss = assume_cross_entropy_loss
        if self.assume_cross_entropy_loss:
            print('Running while assuming cross entropy loss')
        else:
            print('Running without assuming cross entropy loss')

    def feed_forward(self, X_batch):
        if self.first_feed_forward:
            self.first_feed_forward = False
            self.batch_size = len(X_batch)
            self.idx_batch_size = range(self.batch_size)

        self.input = X_batch
        input_dot_weights = self.input.reshape(self.batch_size, -1).dot(
            self.weights)

        p_un = np.exp(input_dot_weights)
        self.output = p_un / p_un.sum(1)[:, None]
        return self.output

    def back_prop(self, loss_derivative_output=None):
        if self.first_back_prop:
            self.first_back_prop = False
            self.d = np.zeros(shape=(self.batch_size, self.num_categories))

        self.d[:] = 0
        self.d[self.idx_batch_size, y_batch] = 1

        if not self.assume_cross_entropy_loss:

            s = loss_derivative_output * self.output
            ct2 = s - s.sum(1)[:, None] * self.output
            loss_derivative_weights = (
                    ct2[:, None, :] * self.input.reshape(batch_dl1_altsize, -1)[
                                      :, :,
                                      None]).sum(0)
            self.loss_derivative_weights = loss_derivative_weights

            loss_derivative_input = ct2.dot(self.weights.T)
            loss_derivative_input = loss_derivative_input.reshape(batch_size,
                                                                  input_image_size,
                                                                  input_image_size,
                                                                  num_filters)

            return loss_derivative_input
        else:
            loss_derivative_weights = (
                    self.input.reshape(batch_size, -1)[:, :, None] \
                    * (self.output - self.d)[:, None, :]).sum(0)
            self.loss_derivative_weights = loss_derivative_weights
            loss_derivative_input = (
                    self.output.dot(self.weights.T) - self.weights[:,
                                                      y_batch].T)
            loss_derivative_input = loss_derivative_input.reshape(batch_size,
                                                                  input_image_size,
                                                                  input_image_size,
                                                                  num_filters)
            return loss_derivative_input


class Model(object):

    def __init__(self):
        self.layers = []
        self.first_run = True

    def feed_forward(self, X_batch):
        if self.first_run:
            self.first_run = False
            self.layers[0].set_first_layer()
            print('Layers are:')
            for l in self.layers:
                print(l, l.weights.shape)

        data = X_batch
        for l in self.layers:
            data = l.feed_forward(data)
        self.output = data

    def loss(self, y_batch):
        self.y_batch = y_batch
        self.batch_size = len(self.y_batch)
        idx_batch_size = range(self.batch_size)
        loss = -np.log(self.output[idx_batch_size, self.y_batch])
        accuracy = self.output.argmax(1) == self.y_batch
        return loss.mean(), accuracy.mean()

    def back_prop(self):
        loss_grad = np.zeros_like(self.output)
        idx_batch_size = range(self.batch_size)
        loss_grad[idx_batch_size, self.y_batch] = -(self.output[
            idx_batch_size, self.y_batch]) ** -1
        for l in self.layers[::-1]:
            loss_grad = l.back_prop(loss_grad)

    def update(self):
        for l in self.layers[::-1]:
            l.update_weights()

    def feed_forward_and_back_prop(self, X_batch, y_batch):
        self.feed_forward(X_batch)
        loss, accuracy = self.loss(y_batch)
        self.back_prop()
        self.update()
        return loss, accuracy


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
#image_size_embedding_size = image_size + K - 1
num_categories = len(set(list(y_train)))

batch_size = 32
num_steps = len(y_train) // batch_size

import time

for num_filters in (10,):
    t0 = time.time()
    print('Training with num_filters ', num_filters)
    np.random.seed(42)
    W1 = np.random.normal(0, 1 / np.sqrt(3 * 3 * input_num_channels),
                          size=(3, 3, input_num_channels, num_filters))
    W2 = np.random.normal(0, 1 / np.sqrt(
        num_filters * input_image_size * input_image_size),
                          size=(num_filters * input_image_size * input_image_size,
                                num_categories))

    m = Model()
    m.layers = [
        Convolution2D(weights=W1, name='Conv1', trainable=True),
        Convolution2D(shape=(5, 5, num_filters, num_filters), trainable=True,
                      name='Conv2'),
        Convolution2D(shape=(3, 3, num_filters, num_filters), trainable=True,
                      name='Conv3'),
        DenseSoftmax(weights=W2, assume_cross_entropy_loss=True,
                     name='DenseSoftmax', trainable=True)
    ]

    for epoch in range(5):

        # Train
        train_loss = averager()
        train_accuracy = averager()
        for i, (X_batch, y_batch) in enumerate(
                batch_generator(X_train, y_train, batch_size, num_steps)):
            if (i + 1) % 10 == 0:
                sys.stdout.write(
                    'Epoch: {} Step {}/{}\r'.format(epoch + 1, i + 1,
                                                    num_steps))
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
            m.feed_forward(X_valid_batch)
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
              ':.2f}, valid acc {:.2f}'.format(
            epoch + 1,
            train_loss,
            train_accuracy,
            valid_loss,
            valid_accuracy
        )
        print(msg)
    t1 = time.time()
    print('Total time', t1 - t0)
