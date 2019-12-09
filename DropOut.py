from Layer import Layer
import numpy as np

class DropOut(Layer):

    def __init__(self, p):
        self.p = p
        self.trainable=False

    def feed_forward(self, x, training, **kwargs):
        if training:
            self.mask=np.random.binomial(1, self.p, size=x.shape)
            np.multiply(x, self.mask, out=x)
        return x

    def back_prop(self, x):
        np.multiply(x, self.mask, out=x)
        return x