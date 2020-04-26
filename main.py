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
        self.x, self.mu, self.var = x, mu, var  # caching
        return out

    def backward(self, dout):
        """
        Shape
            dout: (batch_size, seq_size)
            dx: (batch_size, seq_size)
            dw: (seq_size)
            db: (seq_size)
        """
        dh = np.sqrt(self.var+self.epsilon) * (dout*self.w - (self.x-self.mu) * (self.var+self.epsilon) * np.average((self.x-self.mu) * dout*self.w, axis=0))
        dx = dh - np.average(dh, axis=0)
        dw = np.average((self.x-self.mu) / np.sqrt(self.var+self.epsilon) * dout, axis=0)
        db = np.average(dout, axis=0)
        return dx, dw, db


def main():
    N, S = 32, 124
    x, dout = np.random.randn(N, S), np.random.randn(N, S)
    batch_norm_layer = BatchNorm(S)
    out = batch_norm_layer(x)
    print('# forward')
    print(f' - out.shape: {out.shape}')
    dx, dw, db = batch_norm_layer.backward(dout)
    print('# backward')
    print(f' - dx.shape: {dx.shape}')
    print(f' - dw.shape: {dw.shape}')
    print(f' - db.shape: {db.shape}')


if __name__ == '__main__':
    main()
