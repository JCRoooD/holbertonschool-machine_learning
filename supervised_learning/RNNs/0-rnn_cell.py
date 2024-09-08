#!/usr/bin/env python3
""" RNNCell class module """
import numpy as np


class RNNCell:
    """ RNNCell class """
    def __init__(self, i, h, o):
        """ Class constructor     
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Method that performs forward propagation for one time step
            h_prev: np.ndarray of shape (m, h) containing the previous hidden state
            x_t: np.ndarray of shape (m, i) that contains the data input for the cell
            Returns: h_next, y
                h_next: the next hidden state
                y: the output of the cell
        """
        input_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(input_concat, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y

    def softmax(self, X):
        """ Method that performs softmax activation function
            X: np.ndarray of shape (m, n) containing the input to the softmax
            Returns: np.ndarray of shape (m, n) containing the softmax activation
        """
        return np.exp(X) / (np.sum(np.exp(X), axis=1, keepdims=True))
