import sys
import numpy as np
from scipy.signal import convolve
from tensorflow import keras
from utility_functions import relu,relu_prime, averager,extract_averager_value


def batch_generator(X,y,batch_size,total_count):
    idx=np.arange(0,len(y))
    for i in range(total_count):
        idx_batch=np.random.choice(idx,batch_size)
        yield X[idx_batch],y[idx_batch]

def forward_pass_batch(W1,W2,X_batch,y_batch):
    batch_size=len(y_batch)
    idx_batch_size=range(batch_size)
    l0=X_batch

    for n in range(batch_size):
        for j in range(num_filters):
            l0_conv[n,:,:,j]=convolve(l0[n],W1[::-1,::-1,::-1,j],'same')[:,:,num_channels//2]
    l1=relu(l0_conv)

    l1_dot_W2=l1.reshape(batch_size,-1).dot(W2)

    p_un=np.exp(l1_dot_W2)
    p_sum=p_un.sum(1)
    l2=p_un/p_un.sum(1)[:,None]
    loss=-l1_dot_W2[idx_batch_size,y_batch]+np.log(p_sum)
    accuracy=l2.argmax(1)==y_batch

    return loss.mean(),accuracy.mean()

DATASET='CIFAR100'

if DATASET=='CIFAR10':
    print('Using CIFAR10')
    (X_train_full, y_train_full), (X_test, y_test)  = keras.datasets.cifar10.load_data()
if DATASET=='CIFAR100':
    print('Using CIFAR100')
    (X_train_full, y_train_full), (X_test, y_test)  = keras.datasets.cifar100.load_data()
else:
    print('Using Fashion_MNIST')
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

if len(X_train_full.shape)==3:
    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

y_train=y_train.flatten()
y_valid=y_valid.flatten()
y_test=y_test.flatten()

K=3
num_channels=X_train.shape[3]
image_size=X_train.shape[1]
image_size_embedding_size=image_size+K-1

eta=.001
batch_size=32
idx_batch_size=list(range(batch_size))
num_steps=len(y_train)//batch_size

lt0=np.zeros((batch_size,image_size_embedding_size,image_size_embedding_size,num_channels))
num_categories=len(set(list(y_train)))
for num_filters in (1, 10, 30):

    np.random.seed(42)
    W1 = np.random.normal(0, 1 / np.sqrt(K * K * num_channels),
                          size=(K, K, num_channels, num_filters))
    W2 = np.random.normal(0, 1 / np.sqrt(num_filters * image_size * image_size),
                          size=(num_filters * image_size * image_size, num_categories))

    l0_conv = np.zeros((batch_size, image_size, image_size, num_filters))
    l1 = np.zeros_like(l0_conv)
    f1p = np.zeros_like(l0_conv)

    print('Training with num filters {}'.format(num_filters))

    for epoch in range(5):
        train_loss = averager()
        train_accuracy = averager()

        for i, (X_batch, y_batch) in enumerate(
                batch_generator(X_train, y_train, batch_size, num_steps)):
            if (i + 1) % 10 == 0:
                sys.stdout.write('Epoch: {} Step {}/{}\r'.format(epoch,i+1,num_steps))

            l0 = X_batch
            # lt0=np.zeros((l0.shape[0],l0.shape[1]+K-1,l0.shape[2]+K-1,l0.shape[3]))
            lt0[:] = 0
            lt0[:, K // 2:-K // 2 + 1, K // 2:-K // 2 + 1] = l0

            for n in range(batch_size):
                for j in range(num_filters):
                    l0_conv[n, :, :, j] = convolve(l0[n], W1[::-1, ::-1, ::-1, j], 'same')[:, :,
                                          num_channels // 2]
            l1[:] = 0
            f1p[:] = 0
            l1[:] = relu(l0_conv)
            f1p[:] = relu_prime(l0_conv)

            l1_dot_W2 = l1.reshape(batch_size, -1).dot(W2)

            p_un = np.exp(l1_dot_W2)
            p_sum = p_un.sum(1)
            l2 = p_un / p_un.sum(1)[:, None]
            loss = -l1_dot_W2[idx_batch_size, y_batch] + np.log(p_sum)
            accuracy = l2.argmax(1) == y_batch
            train_loss.send(loss.mean())
            train_accuracy.send(accuracy.mean())

            d = np.zeros(shape=(batch_size, num_categories))
            d[idx_batch_size, y_batch] = 1

            dW2 = (l1.reshape(batch_size, -1)[:, :, None] * (l2 - d)[:, None, :])

            dl1 = (l2.dot(W2.T) - W2[:, y_batch].T).reshape(batch_size, image_size, image_size,
                                                            num_filters)

            dl1_f1p = (dl1 * f1p)

            dW1 = np.array([[(lt0[:, alpha:image_size_embedding_size + alpha - (K - 1),
                              beta:image_size_embedding_size + beta - (K - 1)][:, :, :, :, None] \
                              * dl1_f1p[:, :, :, None, :]).sum((1, 2)) \
                             for beta in range(K)] for alpha in range(K)]).transpose(2, 0, 1, 3, 4)

            W2 += -eta * dW2.sum(0)
            W1 += -eta * dW1.sum(0)

        loss_averager_valid = averager()
        accuracy_averager_valid = averager()

        for X_valid_batch, y_valid_batch in batch_generator(X_valid, y_valid, batch_size,
                                                            len(y_valid) // batch_size):
            loss, accuracy = forward_pass_batch(W1, W2, X_valid_batch, y_valid_batch)
            loss_averager_valid.send(loss)
            accuracy_averager_valid.send(accuracy)

        train_loss, train_accuracy, valid_loss, valid_accuracy = map(extract_averager_value, [
            train_loss,
            train_accuracy,
            loss_averager_valid,
            accuracy_averager_valid]
                                                                     )
        msg = 'Epoch {}: train loss {:.2f}, train acc {:.2f}, valid loss {:.2f}, valid acc {:.2f}'.format(
            epoch + 1,
            train_loss,
            train_accuracy,
            valid_loss,
            valid_accuracy
            )
        print(msg)


