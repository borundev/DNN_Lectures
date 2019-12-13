from Layers.Layer import Layer
import  numpy as np

class MaxPool(Layer):

    def __init__(self, max_pool_size=None):
        if max_pool_size:
            self.max_pool_size_1, self.max_pool_size_2 = max_pool_size
        else:
            self.feed_forward = lambda x: x
            self.back_prop = lambda x: x
        self.trainable=False

    def feed_forward(self, y, training=True):
        mb, n1, n2, ch = y.shape
        if not hasattr(self,
                       'mb') or mb != self.mb or n1 != self.n1 or n2 != self.n2 or ch != self.ch:
            self.mb, self.n1, self.n2, self.ch = mb, n1, n2, ch
            batch = np.array(range(0, mb)) * ch * n1 * n2
            channel = np.array(range(0, ch))
            self.s1 = np.arange(0, n1, self.max_pool_size_1)
            self.s2 = np.arange(0, n2, self.max_pool_size_2)
            start_idx2 = (self.s1[:, None] * n2 + self.s2[None, :]) * ch
            g1 = np.arange(self.max_pool_size_1)
            g2 = np.arange(self.max_pool_size_2)
            grid2 = ((g1[:, None] * n2 + g2[None, :]) * ch).ravel()
            to_take = start_idx2[:, :, None] + grid2[None, None, :]
            self.mb_idx = np.arange(mb)[:, None, None, None]
            self.ch_idx = np.arange(ch)
            self.idxs = [batch[:, None, None, None, None] \
                         + to_take[None, :, :, None, :]
                         + channel[None, None, None, :, None]]
        res = y.take(*self.idxs)
        arg_max_1d = np.argmax(res, axis=4)
        self.row_max_idx = self.s1[None, :, None, None] + arg_max_1d // self.max_pool_size_2
        self.col_max_idx = self.s2[None, None, :, None] + arg_max_1d % self.max_pool_size_2
        return y[self.mb_idx, self.row_max_idx, self.col_max_idx, self.ch_idx]

    def back_prop(self, z):
        yk = np.zeros(shape=(self.mb, self.n1, self.n2, self.ch))
        yk[self.mb_idx, self.row_max_idx, self.col_max_idx, self.ch_idx] = z
        return yk