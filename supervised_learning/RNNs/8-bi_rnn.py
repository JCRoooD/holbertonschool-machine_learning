#!/usr/bin/env python3
"""This moduel contiones the bi_rnn function"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """This function performs forward propagation for a bidirectional RNN
    Args:
        bi_cell: an instance of BidirectionalCell that will be used for the
        forward propagation
        X: numpy.ndarray of shape (t, m, i) that contains the data to be used
            t: the maximum number of time steps
            m: the batch size
            i: the dimensionality of the data
        h_0: numpy.ndarray of shape (m, h) containing the initial hidden state
                m: the batch size
                h: the dimensionality of the hidden state
        h_t: numpy.ndarray of shape (m, h) containing the initial hidden state
        in
        the backward direction
                m: the batch size
                h: the dimensionality of the hidden state
    Returns: H, Y
            H: numpy.ndarray containing all of the concatenated hidden states
            Y: numpy.ndarray containing all of the outputs
    """
    # Extract the shapes of the input data and initial hidden states
    # t: number of time steps, m: batch size, i: input dimensionality
    # h: hidden state dimensionality
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize the arrays to store the hidden states for
    # forward and backward directions
    # Hf will store the forward hidden states, with shape (t + 1, m, h)
    # Hb will store the backward hidden states, with shape (t + 1, m, h)
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    # Set the initial hidden states
    # Hf[0] is set to the initial forward hidden state h_0
    # Hb[-1] is set to the initial backward hidden state h_t
    Hf[0] = h_0
    Hb[-1] = h_t

    # Perform forward propagation through the time steps
    # Iterate over each time step from 0 to t-1
    # Update the forward hidden state at each time step using the
    # forward method of bi_cell
    for i in range(t):
        Hf[i + 1] = bi_cell.forward(Hf[i], X[i])

    # Perform backward propagation through the time steps
    # Iterate over each time step from t-1 to 0
    # Update the backward hidden state at each time step using the
    # backward method of bi_cell
    for i in range(t - 1, -1, -1):
        Hb[i] = bi_cell.backward(Hb[i + 1], X[i])

    H = np.concatenate((Hf[1:], Hb[:t]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
