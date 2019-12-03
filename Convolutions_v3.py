
import sys
if sys.platform=='darwin':
    print('Setting KMP_DUPLICATE_LIB_OK')
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from scipy.signal import convolve2d
from tensorflow import keras
from utility_functions import relu, averager, extract_averager_value, np_random_normal, batch_generator

from Layer import Layer, ActivationFunction
from DenseSoftmax import DenseSoftmax

from Model import Model






class CNN(Layer):

    def __init__(self, weights=None,stride=(1,1),padding=None, **kwargs):
        self.filter_size_1, self.filter_size_2, self.num_channels, self.num_filters = weights.shape
        weights = weights.reshape(-1, self.num_filters)
        self.bias=np.zeros(shape=self.num_filters)
        super().__init__(weights,**kwargs)
        self._make_combined_indx_for_reverse_weights()
        self.stride=stride
        self.padding=padding
        if self.padding:
            self.padding_1,self.padding_2=self.padding
        self.stride_1, self.stride_2 = self.stride

    def __getattr__(self, item):
        if item=='padding_1':
            return self.filter_size_1//2
        elif item=='padding_2':
            return self.filter_size_2//2
        else:
            raise AttributeError('{} has not attribute {}'.format(self.__class__,item))

    def _make_combined_indx_for_reverse_weights(self):
        idx=np.concatenate([np.arange(self.num_channels) + i * self.num_channels for i in
                        range((self.filter_size_1 * self.filter_size_2) - 1, -1, -1)])
        rows = np.concatenate(np.tile(np.split(idx, len(idx) / (self.num_channels)),
                                      self.num_filters))
        cols = np.tile(np.repeat(np.arange(self.num_filters), self.num_channels),
                       self.filter_size_1 * self.filter_size_2)
        self.combined_indx = (rows * self.num_filters + cols).reshape(-1, self.num_channels)

    def _get_reverse_weights(self):
        return self.weights.take(self.combined_indx)

    def _transform(self,x,stride=(1,1),force_padding=False):
        k1, k2 = (self.filter_size_1, self.filter_size_2)
        if not force_padding:
            if not self.padding:
                padding_1 = self.filter_size_1 // 2
                padding_2 = self.filter_size_2 // 2
            else:
                padding_1, padding_2 = self.padding
        else:
            padding_1, padding_2=force_padding
        mb, n1, n2, ch = x.shape
        stride_1,stride_2=stride
        ex1=int((n1+2*padding_1-self.filter_size_1)/stride_1)+1
        ex2 = int((n2 + 2 * padding_2 - self.filter_size_2) / stride_2) + 1
        en1, en2 = n1 + 2 * padding_1, n2 + 2 * padding_2
        y = np.zeros((mb, en1, en2, ch))
        y[:, padding_1:n1 + padding_1, padding_2:n2 + padding_2, :] = x

        s1 = np.arange(en1 - k1 + 1)
        s2 = np.arange(en2 - k2 + 1)
        start_idx2 = (s1[::stride_1, None] * en2 * ch + s2[None,::stride_2] *ch)
        g1 = np.arange(self.filter_size_1)
        g2 = np.arange(self.filter_size_2)
        g3 = np.arange(ch)
        grid3 = (g1[:, None, None] * en2 * ch + g2[None, :, None] *
                 ch + g3[None,None,:]).ravel()
        to_take = start_idx2[:, :, None] + grid3[None, None, :]
        batch = np.array(range(0, mb)) * ch * en1 * en2
        res = y.take(batch[:, None, None, None] + to_take[None, :, :, :])
        return res

    def _transform_back(self,der_y,stride=(1,1)):
        k1, k2 = (self.filter_size_1, self.filter_size_2)

        mb, n1, n2, ch = self.prev_layer.shape

        if not self.padding:
            padding_1 = self.filter_size_1 // 2
            padding_2 = self.filter_size_2 // 2
        else:
            padding_1, padding_2 = self.padding

        stride_1,stride_2=stride


        en1, en2 = n1 + 2 * padding_1, n2 + 2 * padding_2
        #s1 = np.arange(en1 - k1 + 1)
        #s2 = np.arange(en2 - k2 + 1)
        #s1=np.arange(self.filter_size_1 // 2 - padding_1,n1+1,stride_1)
        #s2 = np.arange(self.filter_size_2 // 2 - padding_2, n2+1, stride_2)
        x = np.zeros(shape=(mb,stride_1*der_y.shape[1],stride_2*der_y.shape[2],der_y.shape[3]))

        x[:,::stride_1,::stride_2]=der_y

        return self._transform(x,
                            force_padding=(max(k1 - 1 - padding_1, 0), max(k2 - 1 - padding_2, 0))
                            )[:,:n1,:n2]



    def feed_forward(self, prev_layer):
        if self.first_feed_forward:
            self.first_feed_forward=False
            super().on_first_feed_forward()

        self.prev_layer=prev_layer
        self.prev_layer_transformed=self._transform(prev_layer,self.stride)
        return np.matmul(self.prev_layer_transformed,self.weights) #+ self.bias

    def back_prop(self, next_layer_loss_gradient):

        self.loss_drivative_bias=np.sum(next_layer_loss_gradient,axis=(0,1,2))
        # Do we want sum or average along minibatch direction?
        self.loss_derivative_weights=np.tensordot(self.prev_layer_transformed,
                                                  next_layer_loss_gradient,
                                                  axes=[[0, 1, 2],[0, 1, 2]])

        next_layer_loss_gradient_transformed=self._transform_back(next_layer_loss_gradient,self.stride)
        return np.matmul(next_layer_loss_gradient_transformed,self._get_reverse_weights())

    def update_weights(self):
        if self.trainable:
            #np.add(self.bias,
            #       -self.learning_rate * self.loss_drivative_bias,
            #       out=self.bias
            #       )
            pass
        super().update_weights()

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
        #W11=np.random.normal(0, 1 / np.sqrt(3 * 3 * num_filters),
        #                      size=(3, 3, num_filters, num_filters))
        #W2 = np.random.normal(0, 1 / np.sqrt(
        #    num_filters * input_image_size * input_image_size),
        #                      size=((num_filters * 32 * 32),
        #                            num_categories))

        m = Model()
        learning_rate=0.001
        m.layers = [
            CNN(weights=W1,
                name='Conv1',
                trainable=True,
                stride=(1,1)
                ),
            ActivationFunction(relu),
            #CNN(weights=W11,
            #    name='Conv2',
            #    trainable=True),
            #ActivationFunction(relu),
            #Convolution2D(shape=(5, 5, num_filters, num_filters), trainable=True,
            #              name='Conv2'),
            # Convolution2D(shape=(3, 3, num_filters, num_filters), trainable=True,
            #              name='Conv3'),
            DenseSoftmax(output_dimension=num_categories,
                         name='DenseSoftmax', trainable=True, learning_rate=learning_rate)
        ]

        for epoch in range(10):
            t_start_epoch=time.time()
            # Train
            train_loss = averager()
            train_accuracy = averager()
            for i, (X_batch, y_batch) in enumerate(
                    batch_generator(X_train, y_train, batch_size, num_steps)):
                time_step=time.time()
                if (i + 1) % 10 == 0:
                    delta_time=time_step-t_start_epoch
                    eta=(num_steps/(i+1)-1)*delta_time
                    sys.stdout.write(
                        'Epoch: {} Step {}/{} Time Spent {:.2f}s Estimated Time {:.2f}s\r'.format(epoch + 1, i + 1,
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
                  ':.2f}, valid acc {:.2f}, time taken {:.2f}s'.format(
                epoch + 1,
                train_loss,
                train_accuracy,
                valid_loss,
                valid_accuracy,
                time.time()-t_start_epoch
            )
            print(msg)
        t1 = time.time()
        print('Total time', t1 - t0)
