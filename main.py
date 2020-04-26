#!/usr/bin/env python3
import numpy as np


class BatchNorm:
    def __init__(self, seq_size):
        self.w = np.random.randn(seq_size, seq_size)
        self.b = np.zeros(seq_size)
        self.epsilon = 0.00001
        # todo: backward
        pass

    def __call__(self, args):
        return self.forward(args)

    def forward(self, inp):
        mu = np.average(inp, axis=0)
        var = np.var(inp, axis=0)
        normed_inp = (inp-mu)/np.sqrt(var+self.epsilon)
        out = np.dot(normed_inp, self.w) + self.b
        return out


def main():
    N, S = 32, 124
    inp = np.random.randn(N, S)
    batch_norm_layer = BatchNorm(S)
    out = batch_norm_layer(inp)
    print(out.shape)


if __name__ == '__main__':
    main()
