from Layers.Layer import Layer

class Flatten(Layer):

    def __init__(self,**kwargs):
        self.trainable=False

    def feed_forward(self, prev_layer,**kwargs):
        self.shape=prev_layer.shape
        mb=prev_layer.shape[0]
        return prev_layer.reshape(mb,-1)

    def back_prop(self, next_layer_loss_gradient):
        return next_layer_loss_gradient.reshape(self.shape)