#!/usr/bin/env python3
"""Connvolutional backward prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Performs backward propagation over a convolutional layer of a neural network

    Parameters:
    dZ (numpy.ndarray): Gradient of the cost with respect to the output of the
                        convolutional layer. Shape (m, h, w, c) where m is the
                        number of examples, h and w are the height and width of
                        the output, and c is the number of channels.
    A_prev (numpy.ndarray): Output from the previous layer. Shape (m, h_prev,
                            w_prev, c_prev) where m is the number of examples,
                            h_prev and w_prev are the height and width of the
                            previous layer, and c_prev is the number of channel
    W (numpy.ndarray): Weights for the current layer. Shape (kh, kw, kc, knc)
                       where kh and kw are the height and width of the filter,
                       kc is the number of channels in the previous layer, and
                       knc is the number of filters.
    b (numpy.ndarray): Biases for the current layer.
    padding (str): Type of padding to be used, either 'same' or 'valid'.
    stride (tuple): Tuple of (sh, sw) representing stride height and width.

    Returns:
    dA_prev (numpy.ndarray): Gradient of the cost with respect to the output of
                             the previous layer (A_prev). Shape (m, h_prev, w_prev, c_prev)
    dW (numpy.ndarray): Gradient of the cost with respect to the weights of the
                        current layer (W). Shape (kh, kw, kc, knc)
    db (numpy.ndarray): Gradient of the cost with respect to the biases of the
                       current layer (b). Shape (1, 1, 1, knc)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    m, h, w, c = dZ.shape
    ph, pw = 0, 0
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    if padding == 'same':
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    for i in range(h):
        for j in range(w):
            for k in range(knc):
                dA_prev[:,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :] += W[:, :, :, k] * dZ[:, i, j, k][:, None, None, None]
                for m in range(A_prev.shape[0]):  # iterate over the number of examples
                    dW[:, :, :, k] += A_prev[m,
                                            i * sh: i * sh + kh,
                                            j * sw: j * sw + kw,
                                            :] * dZ[m, i, j, k]
                db[:, :, :, k] += dZ[:, i, j, k]
    return dA_prev, dW, db
