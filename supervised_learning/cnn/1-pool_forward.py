#!/usr/bin/env python3
"""Pooling forward prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    Parameters:
    A_prev (numpy.ndarray): Output from the previous layer. Shape (m, h_prev,
                            w_prev, c_prev) where m is the number of examples,
                            h_prev and w_prev are the height and width of the
                            previous layer, and c_prev is the number of channel
    kernel_shape (tuple): Tuple of (kh, kw) representing kernel height and
                          width.
    stride (tuple): Tuple of (sh, sw) representing stride height and width.
    mode (str): Type of pooling to be used, either 'max' or 'avg'.

    Returns:
    A (numpy.ndarray): The output of the pooling layer. Shape (m, h, w, c)
                       where m is the number of examples, h and w are the
                       height and width of the output, and c is the number of
                       channels.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h = int((h_prev - kh) / sh + 1)
    w = int((w_prev - kw) / sw + 1)
    A = np.zeros((m, h, w, c_prev))
    for i in range(h):
        for j in range(w):
            if mode == 'max':
                A[:, i, j, :] = A_prev[:,
                                       i * sh: i * sh + kh,
                                       j * sw: j * sw + kw,
                                       :].max(axis=(1, 2))
            if mode == 'avg':
                A[:, i, j, :] = A_prev[:,
                                       i * sh: i * sh + kh,
                                       j * sw: j * sw + kw,
                                       :].mean(axis=(1, 2))
    return A
