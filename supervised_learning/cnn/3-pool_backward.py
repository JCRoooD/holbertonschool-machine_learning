#!/usr/bin/env python3
"""Pooling backward prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Performs backward propagation over a pooling layer of a neural network

    Parameters:
    dA (numpy.ndarray): Gradient of the cost with respect to the output of the pooling layer.
                        Shape (m, h_new, w_new, c) where m is the number of examples, h_new
                        and w_new are the height and width of the output, and c is the number
                        of channels.
    A_prev (numpy.ndarray): Output from the previous layer. Shape (m, h_prev, w_prev, c)
                            where m is the number of examples, h_prev and w_prev are the
                            height and width of the previous layer, and c is the number of
                            channels.
    kernel_shape (tuple): Tuple of (kh, kw) representing kernel height and width.
    stride (tuple): Tuple of (sh, sw) representing stride height and width.
    mode (str): Type of pooling used, either 'max' or 'avg'.

    Returns:
    dA_prev (numpy.ndarray): Gradient of the cost with respect to the output of the
                            previous layer (A_prev). Shape (m, h_prev, w_prev, c)
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c):
                if mode == 'max':
                    slice_A_prev = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, k]
                    mask = (slice_A_prev == np.max(slice_A_prev, axis=(1, 2), keepdims=True))
                    dA_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, k] += mask * dA[:, i, j, k, np.newaxis, np.newaxis]
                else:  # mode == 'avg':
                    dA_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, k] += dA[:, i, j, k, np.newaxis, np.newaxis] / (kh * kw)

    return dA_prev
