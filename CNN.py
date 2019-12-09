import numpy as np

from Layer import Layer


class CNN(Layer):

    def __init__(self, weights=None, stride=None, padding=None, bias=False, **kwargs):
        self.filter_size_1, self.filter_size_2, self.num_channels, self.num_filters = weights.shape
        weights = weights.reshape(-1, self.num_filters)
        if bias:
            self.bias = np.zeros(shape=self.num_filters)
        super().__init__(weights, **kwargs)
        self._make_combined_indx_for_reverse_weights()
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
            raise AttributeError('{} has not attribute {}'.format(self.__class__, item))

    def _make_combined_indx_for_reverse_weights(self):
        idx = np.concatenate([np.arange(self.num_channels) + i * self.num_channels for i in
                              range((self.filter_size_1 * self.filter_size_2) - 1, -1, -1)])
        rows = np.concatenate(np.tile(np.split(idx, len(idx) / (self.num_channels)),
                                      self.num_filters))
        cols = np.tile(np.repeat(np.arange(self.num_filters), self.num_channels),
                       self.filter_size_1 * self.filter_size_2)
        self.combined_indx = (rows * self.num_filters + cols).reshape(-1, self.num_channels)

    def _get_reverse_weights(self):
        return self.weights.take(self.combined_indx)

    def _transform(self, x):
        mb, n1, n2, ch = x.shape

        en1, en2 = n1 + 2 * self.padding_1, n2 + 2 * self.padding_2

        # ex1=int((en1-self.filter_size_1)/self.stride_1)+1
        # ex2 = int((en2 - self.filter_size_2) / self.stride_2) + 1

        y = np.zeros((mb, en1, en2, ch))
        y[:, self.padding_1:n1 + self.padding_1, self.padding_2:n2 + self.padding_2, :] = x

        s1 = np.arange(en1 - self.filter_size_1 + 1)
        s2 = np.arange(en2 - self.filter_size_2 + 1)
        start_idx2 = (s1[::self.stride_1, None] * en2 * ch + s2[None, ::self.stride_2] * ch)
        g1 = np.arange(self.filter_size_1)
        g2 = np.arange(self.filter_size_2)
        g3 = np.arange(ch)
        grid3 = (g1[:, None, None] * en2 * ch + g2[None, :, None] *
                 ch + g3[None, None, :]).ravel()
        to_take = start_idx2[:, :, None] + grid3[None, None, :]
        batch = np.array(range(0, mb)) * ch * en1 * en2
        res = y.take(batch[:, None, None, None] + to_take[None, :, :, :])
        return res

    def _transform2(self, x):

        mb, n1, n2, ch = x.shape

        p1_left = self.padding_1 + 1 - self.filter_size_1
        # p1_right = self.padding_1

        p2_left = self.padding_2 + 1 - self.filter_size_2
        # p2_right = self.padding_2

        # d1 = p1_right - p1_left
        # d2 = p2_right - p2_left

        # start position in x
        i1 = max(0, p1_left)
        i2 = max(0, p2_left)

        # start position in y
        # iy1=max(0, -p1_left)
        # iy2=max(0, -p2_left)
        iy1 = i1 - p1_left
        iy2 = i2 - p2_left

        # size of array taken from x
        f1 = x.shape[1] - i1
        f2 = x.shape[2] - i2
        y = np.zeros(shape=(x.shape[0],
                            x.shape[1] + self.filter_size_1 - 1,
                            x.shape[2] + self.filter_size_2 - 1,
                            x.shape[3])
                     )
        y[:,
        iy1:iy1 + f1,
        iy2:iy2 + f2
        ] = x[:, i1:, i2:, :]

        en1, en2 = y.shape[1], y.shape[2]

        s1 = np.arange(en1 - self.filter_size_1 + 1)
        s2 = np.arange(en2 - self.filter_size_2 + 1)
        start_idx2 = (s1[:, None] * en2 * ch + s2[None, :] * ch)
        g1 = np.arange(self.filter_size_1)
        g2 = np.arange(self.filter_size_2)
        g3 = np.arange(ch)
        grid3 = (g1[:, None, None] * en2 * ch + g2[None, :, None] *
                 ch + g3[None, None, :]).ravel()
        to_take = start_idx2[:, :, None] + grid3[None, None, :]
        batch = np.array(range(0, mb)) * ch * en1 * en2
        res = y.take(batch[:, None, None, None] + to_take[None, :, :, :])
        return res

    def _transform_back(self, der_y):
        mb, n1, n2, ch = self.prev_layer.shape

        x = np.zeros(shape=(mb,
                            max(self.stride_1 * der_y.shape[1], n1),
                            max(self.stride_2 * der_y.shape[2], n2),
                            der_y.shape[3]))

        x[:, :self.stride_1 * der_y.shape[1]:self.stride_1,
        :self.stride_2 * der_y.shape[2]:self.stride_2] = der_y
        return self._transform2(x)[:, :n1, :n2]

    def feed_forward(self, prev_layer, **kwargs):
        if self.first_feed_forward:
            self.first_feed_forward = False
            super().on_first_feed_forward()

        self.prev_layer = prev_layer
        self.prev_layer_transformed = self._transform(prev_layer)
        res = np.matmul(self.prev_layer_transformed, self.weights)
        if hasattr(self, 'bias'):
            np.add(res, self.bias, out=res)
        return res

    def back_prop(self, next_layer_loss_gradient):

        self.loss_drivative_bias = np.sum(next_layer_loss_gradient, axis=(0, 1, 2))
        # Do we want sum or average along minibatch direction?
        self.loss_derivative_weights = np.tensordot(self.prev_layer_transformed,
                                                    next_layer_loss_gradient,
                                                    axes=[[0, 1, 2], [0, 1, 2]])

        next_layer_loss_gradient_transformed = self._transform_back(next_layer_loss_gradient)
        return np.matmul(next_layer_loss_gradient_transformed, self._get_reverse_weights())
