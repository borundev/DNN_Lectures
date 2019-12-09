from utility_functions import np_random_normal
from Layer import Layer
import numpy as np

class DenseSoftmax(Layer):

    def __init__(self, weights=None, shape=None, output_dimension=None, **kwargs):

        if shape is not None:
            weights = np_random_normal(0, 1 / np.sqrt(shape[0]), size=shape)

        super().__init__(weights, **kwargs)
        self.output_dimension=output_dimension
        self.num_categories = self.weights.shape[-1] if self.weights is not None else output_dimension
        self.batch_size = None

    def feed_forward(self, X_batch, **kwargs):
        if self.first_feed_forward:
            self.first_feed_forward = False
            self.batch_size = len(X_batch)
            self.idx_batch_size = range(self.batch_size)
            if self.weights is None:
                print("initiating")
                shape=(X_batch.shape[1],self.output_dimension)
                self.weights=np_random_normal(0,1/np.sqrt(shape[0]),size=shape)
            super().on_first_feed_forward()

        self.input = X_batch
        input_dot_weights = self.input.dot(self.weights)

        p_un = np.exp(input_dot_weights)
        self.output = p_un / p_un.sum(1)[:, None]
        return self.output

    def back_prop(self, loss_derivative_output=None):
        if self.first_back_prop:
            self.first_back_prop = False



        s = loss_derivative_output * self.output
        ct2 = s - s.sum(1)[:, None] * self.output
        loss_derivative_weights = (
                ct2[:, None, :] * self.input.reshape(self.batch_size, -1)[
                                  :, :,
                                  None]).sum(0)
        self.loss_derivative_weights = loss_derivative_weights

        loss_derivative_input = ct2.dot(self.weights.T)

        return loss_derivative_input
