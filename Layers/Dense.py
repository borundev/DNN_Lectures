from Layer import Layer
from utility_functions import np_random_normal
import numpy as np

class Dense(Layer):

    def __init__(self,weights=None, output_dimension=None, bias=True, **kwargs):
        if weights is not None:
            self.output_dimension = weights[1]
        else:
            self.output_dimension=output_dimension
        if bias:
            self.bias=np.zeros(output_dimension)
        super().__init__(weights, **kwargs)


    def feed_forward(self, prev_layer, **kwargs):
        if self.first_feed_forward:
            if self.weights is None:
                shape=(prev_layer.shape[-1], self.output_dimension)
                self.weights=np_random_normal(0,1/np.sqrt(shape[0]),size=shape)
            super().on_first_feed_forward()
        self.input=prev_layer
        res=np.matmul(self.input, self.weights)
        # why is dot so much slower than matmul?
        #res=self.input.dot(self.weights)
        if hasattr(self, 'bias'):
            np.add(res, self.bias,out=res)
        return res

    def back_prop(self, next_layer_loss_gradient):
        if not hasattr(self,'axis_to_sum_over'):
            self.axis_to_sum_over=tuple(range(len(next_layer_loss_gradient.shape)-1))
        self.loss_derivative_weights=np.tensordot(self.input, next_layer_loss_gradient,
                                                  [self.axis_to_sum_over,self.axis_to_sum_over])
        if hasattr(self,'bias'):
            self.loss_drivative_bias = np.sum(next_layer_loss_gradient,axis=self.axis_to_sum_over)
        if not self.first_layer:
            return next_layer_loss_gradient.dot(self.weights.T)