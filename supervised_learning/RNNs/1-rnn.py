#!/usr/bin/env python3
""" RNN module """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ Function that performs forward propagation for a simple RNN
        rnn_cell: instance of RNNCell that will be used for the forward propagation
        X: np.ndarray of shape (t, m, i) that contains the data input for the RNN
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: np.ndarray of shape (m, h) containing the initial hidden state
            h: dimensionality of the hidden state
        Returns: H, Y
            H: np.ndarray containing all of the hidden states
            Y: np.ndarray containing all of the outputs
    """
    t = X.shape[0]
    m = X.shape[1]
    
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    for i in range(t):
        if i == 0:
            H[i] = h_0
        H[i + 1], Y[i] = rnn_cell.forward(H[i], X[i])
    return H, Y
