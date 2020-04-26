#!/usr/bin/env python3
import numpy as np


class BatchNorm:
    def __init__(self, seq_size):
        # params
        self.w = np.random.randn(seq_size)
        self.b = np.zeros(seq_size)
        self.epsilon = 0.00001
        # cache
        self.x, self.mu, self.var = None, None, None

    def __call__(self, args):
        return self.forward(args)

    def forward(self, x):
        """
        Shape
            x: (batch_size, seq_size)
            out: (batch_size, seq_size)
        """
        mu = np.average(x, axis=0)
        var = np.var(x, axis=0)
        normlized_x = (x-mu) / np.sqrt(var+self.epsilon)
        out = normlized_x*self.w + self.b
        return out

    def backward(self, dout):
        """
        Shape
            dout: (batch_size, seq_size)
            dx: (batch_size, seq_size)
            dw: (seq_size)
            db: (seq_size)
        """
        pass


def main():
    N, S = 32, 124
    x = np.random.randn(N, S)
    batch_norm_layer = BatchNorm(S)
    out = batch_norm_layer(x)
    print(out.shape)
    print(out)


if __name__ == '__main__':
    main()
