#!/usr/bin/env python3
"""Connvolutional forward prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

    Parameters:
    A_prev (numpy.ndarray): Output from the previous layer. Shape (m, h_prev,
                            w_prev, c_prev) where m is the number of examples,
                            h_prev and w_prev are the height and width of the
                            previous layer, and c_prev is the number of channel
    W (numpy.ndarray): Weights for the current layer. Shape (kh, kw, kc, knc)
                       where kh and kw are the height and width of the filter,
                       kc is the number of channels in the previous layer, and
                       knc is the number of filters.
    b (numpy.ndarray): Biases for the current layer.
    activation (function): Activation function to be used.
    padding (str): Type of padding to be used, either 'same' or 'valid'.
    stride (tuple): Tuple of (sh, sw) representing stride height and width.

    Returns:
    Z (numpy.ndarray): The output of the convolutional layer. Shape (m, h, w,
                       knc) where m is the number of examples, h and w are the
                       height and width of the output, and knc is the number of
                       filters.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    if padding == 'same':
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    h = int((h_prev + 2 * ph - kh) / sh + 1)
    w = int((w_prev + 2 * pw - kw) / sw + 1)
    Z = np.zeros((m, h, w, knc))
    for i in range(h):
        for j in range(w):
            for k in range(knc):
                Z[:, i, j, k] = (W[:, :, :, k] *
                                 A_prev[:,
                                        i * sh: i * sh + kh,
                                        j * sw: j * sw + kw,
                                        :]).sum(axis=(1, 2, 3))
    return activation(Z + b)
