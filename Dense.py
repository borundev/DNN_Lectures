from Layer import Layer
from utility_functions import np_random_normal
import numpy as np

class Dense(Layer):

    def __init__(self,weights=None, output_dimension=None, bias=True):
        super().__init__(weights)
        if self.weights is not None:
            self.output_dimension = self.weights[1]
        else:
            self.output_dimension=output_dimension
        if bias:
            self.bias=np.zeros(output_dimension)

    def feed_forward(self, prev_layer, **kwargs):
        if self.first_feed_forward:
            if self.weights is None:
                shape=(prev_layer.shape[1], self.output_dimension)
                self.weights=np_random_normal(0,1/np.sqrt(shape[0]),size=shape)
            super().on_first_feed_forward()
        self.input=prev_layer
        res=self.input.dot(self.weights)
        if hasattr(self, 'bias'):
            np.add(res, self.bias, out=res)
        return res

    def back_prop(self, next_layer_loss_gradient):
        self.loss_derivative_weights=np.tensordot(self.input, next_layer_loss_gradient, (0, 0))
        if hasattr(self,'bias'):
            self.loss_drivative_bias = next_layer_loss_gradient.sum(0)
        return next_layer_loss_gradient.dot(self.weights.T)