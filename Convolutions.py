import sys

import numpy as np
from scipy.signal import convolve
from tensorflow import keras
from utility_functions import relu, relu_prime, averager, extract_averager_value


def batch_generator(X, y, batch_size, total_count):
    idx = np.arange(0, len(y))
    for i in range(total_count):
        idx_batch = np.random.choice(idx, batch_size)
        yield X[idx_batch], y[idx_batch]


class Layer(object):

    def __init__(self,weights,learning_rate=0.001):
        self.learning_rate = learning_rate
        self.weights=weights
        self.first_feed_forward=True
        self.first_back_prop=True

    def feed_forward(self, prev_layer):
        raise NotImplementedError

    def back_prop(self, next_layer_loss_gradient):
        raise NotImplementedError

    def update_weights(self):
        np.add(self.weights,-self.learning_rate * self.d_weights, out=self.weights)


class Convolution2D(Layer):

    def __init__(self, weights=None, shape=None):

        if weights is None:
            assert shape is not None, 'Both weights and shape cannot be None'
            self.filter_size, _, self.num_channels, self.num_filters = shape
            weights = np.random.normal(0, 1 / np.sqrt(
                self.filter_size * self.filter_size * self.num_channels),
                                       size=(self.filter_size,
                                                  self.filter_size,
                                                  self.num_channels,
                                                  self.num_filters))

        super().__init__(weights)
        self.filter_size, _, self.num_channels, self.num_filters = self.weights.shape
        self.batch_size = None
        self.image_size_embedding_size = None

    def feed_forward(self, X_batch):
        if self.first_feed_forward: # First run
            self.first_feed_forward=False
            self.batch_size = len(X_batch)
            self.image_size = X_batch.shape[1]
            self.l0_conv = np.zeros((self.batch_size, self.image_size,
                                     self.image_size,
                                     self.num_filters))
            self.l1 = np.zeros_like(self.l0_conv)
            self.l1p = np.zeros_like(self.l0_conv)

        self.l0 = X_batch

        for n in range(self.batch_size):
            for j in range(self.num_filters):
                self.l0_conv[n, :, :, j] = convolve(self.l0[n],
                                                    self.weights[::-1, ::-1,
                                                    ::-1, j],
                                                    'same')[:, :,
                                           self.num_channels
                                           // 2]
        self.l1[:] = relu(self.l0_conv)
        self.l1p[:] = relu_prime(self.l0_conv)
        return self.l1

    def back_prop(self, dl1):
        if self.first_back_prop:
            self.first_back_prop=False
            self.image_size_embedding_size = self.image_size + self.filter_size - 1
            self.lt0 = np.zeros((self.batch_size,
                                 self.image_size_embedding_size,
                                 self.image_size_embedding_size,
                                 self.num_channels))
            self.dl1_l1p = np.zeros_like(self.l1p)

        np.multiply(dl1,self.l1p,out=self.dl1_l1p)
        self.lt0[:, self.filter_size // 2:-self.filter_size // 2 + 1, self.filter_size // 2:-self.filter_size // 2 + 1] \
            = self.l0
        self.d_weights = np.array(
            [[(self.lt0[:, alpha:image_size_embedding_size + alpha - (self.filter_size - 1),
               beta:image_size_embedding_size + beta - (self.filter_size - 1)][:, :, :,
               :, None] \
               * self.dl1_l1p[:, :, :, None, :]).sum((1, 2)) \
              for beta in range(self.filter_size)] for alpha in range(
                self.filter_size)]).transpose(2, 0, 1, 3, 4).sum(0)

        return None


class DenseSoftmax(Layer):

    def __init__(self, weights=None, shape=None,assume_cross_entropy_loss=True):

        if weights is None:
            assert shape is not None, 'Both weights and shape cannot be None'
            weights = np.random.normal(0, 1, size=shape)

        super().__init__(weights)
        self.num_categories=self.weights.shape[1]
        self.batch_size = None
        self.assume_cross_entropy_loss=assume_cross_entropy_loss
        if self.assume_cross_entropy_loss:
            print('Running while assuming cross entropy loss')
        else:
            print('Running without assuming cross entropy loss')

    def feed_forward(self, X_batch):
        if self.first_feed_forward:
            self.first_feed_forward=False
            self.batch_size = len(X_batch)
            self.idx_batch_size = range(self.batch_size)

        self.l1 = X_batch
        l1_dot_W2 = self.l1.reshape(self.batch_size, -1).dot(self.weights)

        p_un = np.exp(l1_dot_W2)
        self.l2 = p_un / p_un.sum(1)[:, None]
        return self.l2

    def back_prop(self, next_layer_loss_gradient=None):
        if self.first_back_prop:
            self.first_back_prop=False
            self.d = np.zeros(shape=(self.batch_size, self.num_categories))

        self.d[:]=0
        self.d[self.idx_batch_size, y_batch] = 1

        if not self.assume_cross_entropy_loss:

            s=next_layer_loss_gradient*self.l2
            ct2=s-s.sum(1)[:,None]*self.l2
            d_weights_alt=(ct2[:,None,:]*self.l1.reshape(batch_size,-1)[:,:,None]).sum(0)
            self.d_weights=d_weights_alt

            dl1_alt=ct2.dot(self.weights.T)
            dl1_alt=dl1_alt.reshape(batch_size,image_size,image_size,num_filters)

            return dl1_alt
        else:
            d_weights = (self.l1.reshape(batch_size, -1)[:, :, None] \
                         * (self.l2 - self.d)[:, None, :]).sum(0)
            self.d_weights=d_weights
            dl1 = (self.l2.dot(self.weights.T) - self.weights[:,y_batch].T)
            dl1=dl1.reshape(batch_size,image_size,image_size,num_filters)
            return dl1


class Model(object):

    def __init__(self):
        self.layers = []

    def feed_forward(self, X_batch):
        data = X_batch.copy()
        for l in self.layers:
            data = l.feed_forward(data)
        self.output = data

    def loss(self, y_batch):
        self.y_batch=y_batch
        self.batch_size = len(self.y_batch)
        idx_batch_size = range(self.batch_size)
        loss = -np.log(self.output[idx_batch_size, self.y_batch])
        accuracy = self.output.argmax(1) == self.y_batch
        return loss.mean(), accuracy.mean()

    def back_prop(self):
        loss_grad = np.zeros_like(self.output)
        idx_batch_size = range(self.batch_size)
        loss_grad[idx_batch_size, self.y_batch]=-(self.output[idx_batch_size,self.y_batch])**-1
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


DATASET = 'Fashion_MNIST'

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

K = 3
num_channels = X_train.shape[3]
image_size = X_train.shape[1]
image_size_embedding_size = image_size + K - 1
num_categories = len(set(list(y_train)))

batch_size = 32
num_steps = len(y_train) // batch_size

import time

for num_filters in (10,):
    t0 = time.time()
    print('Training with num_filters ', num_filters)
    np.random.seed(42)
    W1 = np.random.normal(0, 1 / np.sqrt(K * K * num_channels),
                          size=(K, K, num_channels, num_filters))
    W2 = np.random.normal(0, 1 / np.sqrt(
        num_filters * image_size * image_size),
                          size=(num_filters * image_size * image_size,
                                num_categories))

    m = Model()
    m.layers = [
        Convolution2D(weights=W1),
        DenseSoftmax(weights=W2, assume_cross_entropy_loss=True)
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
