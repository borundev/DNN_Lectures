import numpy as np
from Layers.Dense import Dense

from Layers.Layer import Layer


class CNN(Layer):

    def __init__(self, weights=None,
                 num_filters=None,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 bias=False, **kwargs):

        if weights is not None:
            self.filter_size_1, self.filter_size_2, self.num_channels, self.num_filters = weights.shape
            kernel_size=self.filter_size_1, self.filter_size_2

            weights = weights.reshape(-1, self.num_filters)
            self._make_combined_indx_for_reverse_weights()
            self.dense=Dense(weights=weights,bias=bias, first_layer=True)
        else:
            self.dense=Dense(output_dimension=num_filters,bias=bias, first_layer=True)
        super().__init__(weights=None, **kwargs)


        if num_filters:
            self.num_filters=num_filters

        if kernel_size:
            if type(kernel_size) in [list,tuple]:
                self.filter_size_1, self.filter_size_2 = kernel_size
            else:
                self.filter_size_1=self.filter_size_2=kernel_size
        else:
            raise ValueError('kernel size must be given either expliitely or implicitely as the '
                             'shape of weight tensor')
        if padding:
            self.padding_1, self.padding_2 = padding
        if stride:
            self.stride_1, self.stride_2 = stride

    def __getattr__(self, item):
        if item == 'padding_1':
            return self.filter_size_1 // 2
        elif item == 'padding_2':
            return self.filter_size_2 // 2
        elif item in ('stride_1', 'stride_2'):
            return 1
        else:
            raise AttributeError('{} does not have attribute {}'.format(self.__class__, item))

    def _make_combined_indx_for_reverse_weights(self):
        idx = np.concatenate([np.arange(self.num_channels) + i * self.num_channels for i in
                              range((self.filter_size_1 * self.filter_size_2) - 1, -1, -1)])
        rows = np.concatenate(np.tile(np.split(idx, len(idx) / (self.num_channels)),
                                      self.num_filters))
        cols = np.tile(np.repeat(np.arange(self.num_filters), self.num_channels),
                       self.filter_size_1 * self.filter_size_2)
        self.combined_indx = (rows * self.num_filters + cols).reshape(-1, self.num_channels)

    def _get_reverse_weights(self):
        return self.dense.weights.take(self.combined_indx)

    def _transform(self, x):
        mb, n1, n2, ch = x.shape
        en1, en2 = n1 + 2 * self.padding_1, n2 + 2 * self.padding_2
        y = np.zeros((mb, en1, en2, ch))
        y[:, self.padding_1:n1 + self.padding_1, self.padding_2:n2 + self.padding_2, :] = x
        return self._take(y,(self.stride_1,self.stride_2))


    def _take(self,y,stride=(1,1)):
        """
        Takes a 4 dimensional array that is an image with the indices being the minibatch, row,
        column and channel. It returns a 4 dimensional array where the first 3 indices are the same
        but the 4th containes all the elements in a cuboid that will get multiplied with flattened
        convolution filters defined on the respective channels.


        :param y:
        :param stride:
        :return:
        """
        stride_1,stride_2=stride
        mb, en1, en2, ch = y.shape

        # Till we discuss the minibatch index, all comments are for the first image

        # Make a 2d array of indices of the top-left edges of the windows from which to
        # take elements. These are to be the indices on the first channel. This makes the indices
        # the top-left-back end of the cuboid to be taken
        s1 = np.arange(0,en1 - self.filter_size_1 + 1, stride_1)
        s2 = np.arange(0,en2 - self.filter_size_2 + 1, stride_2)
        start_idx = (s1[:, None] * en2 * ch + s2[None, :] * ch)

        # Make a grid of elements to be taken in the entire cuboid whose top-left-back
        # indices we have taken above. This is done only for the first of the above cuboids in mind.
        # Note the cuboid elements are flattened and will now be along the 4th direction of the
        # output
        g1 = np.arange(self.filter_size_1)
        g2 = np.arange(self.filter_size_2)
        g3 = np.arange(ch)
        grid = (g1[:, None, None] * en2 * ch + g2[None, :, None] *
                 ch + g3[None, None, :]).ravel()

        # Combine the above two to make a 3d array which corresponds to just the first image in a
        # minibatch.
        grid_to_take = start_idx[:, :, None] + grid[None, None, :]

        # Make and index for the starting entry in every image in a minibatch
        batch = np.array(range(0, mb)) * ch * en1 * en2

        # This is the final result
        res = y.take(batch[:, None, None, None] + grid_to_take[None, :, :, :])
        return res

    def _transform_back(self, der_y):
        mb, n1, n2, _ = self.prev_layer.shape
        ch = der_y.shape[3]

        m1= max(self.stride_1 * der_y.shape[1], n1)
        m2 = max(self.stride_2 * der_y.shape[2], n2)

        z = np.zeros(shape=(mb,m1,m2,ch))
        y = np.zeros(shape=(mb, m1 + self.filter_size_1 - 1, m2 + self.filter_size_2 - 1,ch))

        z[:,
            :self.stride_1 * der_y.shape[1]:self.stride_1,
            :self.stride_2 * der_y.shape[2]:self.stride_2
        ] = der_y


        p1_left = self.padding_1 + 1 - self.filter_size_1
        p2_left = self.padding_2 + 1 - self.filter_size_2

        # i1,i2 are the start positions in z and iy1,iy2 are the start positions in y
        i1=i2=iy1=iy2=0

        if p1_left>0:
            i1 = p1_left
        else:
            iy1 = -p1_left

        if p2_left>0:
            i2 = p2_left
        else:
            iy2 = -p2_left

        # size of array taken from x
        f1 = z.shape[1] - i1
        f2 = z.shape[2] - i2

        y[:,
            iy1:iy1 + f1,
            iy2:iy2 + f2
        ] = z[:, i1:, i2:, :]

        return self._take(y)[:,:n1,:n2]


    def feed_forward(self, prev_layer, **kwargs):

        self.prev_layer = prev_layer
        self.prev_layer_transformed = self._transform(prev_layer)
        res=self.dense.feed_forward(self.prev_layer_transformed)
        if self.first_feed_forward:
            self.num_channels=prev_layer.shape[3]
            self._make_combined_indx_for_reverse_weights()
            super().on_first_feed_forward()

        return res

    def back_prop(self, next_layer_loss_gradient):

        self.dense.back_prop(next_layer_loss_gradient)

        next_layer_loss_gradient_transformed = self._transform_back(next_layer_loss_gradient)
        return np.matmul(next_layer_loss_gradient_transformed, self._get_reverse_weights())

    def update_weights(self):
        self.dense.update_weights()
