# -*- coding: utf-8 -*-
"""
Created on 2018/7/7

@author: Vasili
"""
import numpy as np


class BatchGenerator(object):
    def __init__(self, tensor_in, tensor_out, batch_size, seq_length):
        """

        :param tensor_in:
        :param tensor_out:
        :param batch_size:
        :param seq_length:
        """
        self.tensor_in = tensor_in
        self.tensor_out = tensor_out

        self.batch_size = batch_size
        self.seq_length = seq_length

    def reset(self):
        self.point = 0

    def create_batches(self):
        self.num_batches = int(self.tensor_in.size() / (self.batch_size * self.seq_length))
        self.tensor_in = self.tensor_in[:self.num_batches * self.batch_size * self.seq_length]  # 目的是为了取整
        self.tensor_out = self.tensor_out[:self.num_batches * self.batch_size * self.seq_length]

        if self.num_batches == 0:
            assert False, "No enough data."

        self.x_batches = np.split(self.tensor_in.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(self.tensor_out.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.point], self.y_batches[self.point]
        self.point += 1
        return x, y


class CopyBatchGenerator(BatchGenerator):

    def __init__(self, data, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        tensor_in = data
        tensor_out = np.copy(tensor_in)  # 拷贝的目的是修改副本，主的不会被影响
        tensor_out[:-1] = tensor_in[1:]
        tensor_out[-1] = tensor_in[0]

        super(CopyBatchGenerator, self).__init__(tensor_in, tensor_out, batch_size, seq_length)


class PreBatchGenerator(BatchGenerator):
    def __init__(self, data_in, data_out, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length

        tensor_in = np.array(data_in)
        tensor_out = np.array(data_out)

        super(PreBatchGenerator, self).__init__(tensor_in, tensor_out, batch_size, seq_length)


if __name__ == "__main__":
    # [2,3,4]
    a = np.array([i for i in range(24)])
    batch = np.split(a.reshape(3, -1), 2, 1)
