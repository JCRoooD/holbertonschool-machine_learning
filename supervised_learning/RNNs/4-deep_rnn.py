#!/usr/bin/env python3
"""This module contains the function that performs
forward propagation on a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """This function performs forward propagation on a deep RNN
    Args:
        rnn_cells: list of RNNCell instances of length l that will be used
                   for the forward propagation
                   l: is the number of layers
        X: numpy.ndarray of shape (t, m, i) that contains the input data
           t: is the maximum number of time steps
           m: is the batch size
           i: is the dimensionality of the data
        h_0: numpy.ndarray of shape (l, m, h) that contains the initial hidden
             state
             h: is the dimensionality of the hidden state
    Returns: H, Y
             H: numpy.ndarray containing all of the hidden states
             Y: numpy.ndarray containing all of the outputs
    """
    # Extract the inputs
    t, m, i = X.shape
    h = h_0.shape[-1]
    length = len(rnn_cells)

    H = np.zeros((t + 1, length, m, h))
    # print("-"*50)

    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    for i in range(t):
        # calculate the hidden state for each cell
        for j in range(length):
            if i == 0:
                H[i, j] = h_0[j]

            if j == 0:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j], X[i])

            else:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(
                    H[i, j], H[i + 1, j - 1])

    return H, Y
