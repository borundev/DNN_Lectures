import sys

from Dense import Dense
from Flatten import Flatten
from Softmax import Softmax

if sys.platform == 'darwin':
    import os

    if os.environ.get('KMP_DUPLICATE_LIB_OK', 'False') != 'True':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        print('Setting KMP_DUPLICATE_LIB_OK')

import numpy as np
from tensorflow import keras
from utility_functions import relu, averager, extract_averager_value, np_random_normal, \
    batch_generator

from Layer import ActivationFunction
from Model import Model

from CNN import CNN

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
        if len(sys.argv) == 1 or sys.argv[1] != 'tf':
            np.random.seed(42)
            t0 = time.time()
            print('Training with num_filters ', num_filters)
            W1 = np_random_normal(0, 1 / np.sqrt(3 * 3 * input_num_channels),
                                  size=(3, 3, input_num_channels, num_filters))
            # W2 = np.random.normal(0, 1 / np.sqrt(
            #    num_filters * input_image_size * input_image_size),
            #                      size=((num_filters * 32 * 32),
            #                            num_categories))

            m = Model()
            learning_rate = 0.001
            m.layers = [
                CNN(weights=W1,
                    name='Conv1',
                    trainable=True,
                    stride=(1, 1),
                    ),
                ActivationFunction(relu),
                # MaxPool((2,4)),
                # CNN(num_filters=4, kernel_size=5, stride=(2,2)),
                # ActivationFunction(relu),
                # DropOut(0.5),
                Flatten(),
                Dense(output_dimension=num_categories),
                Softmax()
                # DenseSoftmax(output_dimension=num_categories,
                #             name='DenseSoftmax', trainable=True, learning_rate=learning_rate)
                ]

            for epoch in range(10):
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
                            'Epoch: {} Step {}/{} Time Spent {:.2f}s Estimated Time {'
                            ':.2f}s\r'.format(
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
                    m.feed_forward(X_valid_batch, training=False)
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
        else:
            print('Using tensorflow')
            from functools import partial

            DefaultConv2D = partial(keras.layers.Conv2D,
                                    kernel_size=3, activation='relu', padding="SAME")

            model = keras.models.Sequential([
                DefaultConv2D(filters=num_filters, kernel_size=3, input_shape=[32, 32, 3]),
                keras.layers.MaxPooling2D(pool_size=2),
                # DefaultConv2D(filters=128),
                # DefaultConv2D(filters=128),
                # keras.layers.MaxPooling2D(pool_size=2),
                # DefaultConv2D(filters=256),
                # DefaultConv2D(filters=256),
                # keras.layers.MaxPooling2D(pool_size=2),
                # keras.layers.Flatten(),
                # keras.layers.Dense(units=128, activation='relu'),
                keras.layers.Dropout(0.5),
                # keras.layers.Dense(units=64, activation='relu'),
                # keras.layers.Dropout(0.5),
                keras.layers.Flatten(),
                keras.layers.Dense(units=10, activation='softmax'),
                ])
            t0 = time.time()
            model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
                          metrics=["accuracy"])
            history = model.fit(X_train, y_train, epochs=10, validation_data=[X_valid, y_valid])
            t1 = time.time()
            print('Total time', t1 - t0)
