import numpy as np


class Layer(object):

    def __init__(self, weights, trainable=True, learning_rate=0.001, name=None, first_layer=False):
        self.learning_rate = learning_rate
        self.weights = weights.copy() if weights is not None else None
        self.first_feed_forward = True
        self.first_back_prop = True
        self.first_layer = first_layer
        self.trainable = trainable
        self.name = name

    def feed_forward(self, prev_layer,**kwargs):
        raise NotImplementedError

    def on_first_feed_forward(self):
        pass

    def back_prop(self, next_layer_loss_gradient):
        raise NotImplementedError

    def update_weights(self):
        if self.trainable:
            np.add(self.weights,
                   -self.learning_rate * self.loss_derivative_weights,
                   out=self.weights)
            if hasattr(self, 'bias'):
                np.add(self.bias,
                       -self.learning_rate * self.loss_drivative_bias,
                       out=self.bias
                       )


class ActivationFunction(Layer):

    def __init__(self, activation_function):
        self.activation_function = activation_function
        self.trainable = False

    def feed_forward(self, prev_layer, **kwargs):
        self.prev_layer = prev_layer
        return self.activation_function(prev_layer)

    def back_prop(self, next_layer_loss_gradient):
        return next_layer_loss_gradient * self.activation_function(self.prev_layer, der=True)
