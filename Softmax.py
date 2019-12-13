from Layers.Layer import Layer
import numpy as np

class Softmax(Layer):

    def __init__(self):
        self.trainable=False

    def feed_forward(self, prev_layer,**kwargs):
        p_un = np.exp(prev_layer)
        self.output = p_un / p_un.sum(1)[:, None]
        return self.output

    def back_prop(self, next_layer_loss_gradient):
        s = next_layer_loss_gradient * self.output
        return s - self.output*s.sum(1)[:,np.newaxis]