import numpy as np


# TODO: This only handle categorical cross entropy in loss and backprop (the first
# step  uses loss derivative assuming categorical cross entropy)
# but can easily be extended to use others

class Model(object):

    def __init__(self):
        self.layers = []
        self.first_run = True

    def feed_forward(self, X_batch):
        if self.first_run:
            self.first_run = False
            #self.layers[0].set_first_layer()

        data = X_batch
        for l in self.layers:
            data = l.feed_forward(data)
        self.output = data
        return self.output

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
