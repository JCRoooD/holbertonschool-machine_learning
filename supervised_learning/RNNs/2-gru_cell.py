#!/usr/bin/env python3
"""This module for GRUCell class"""
import numpy as np


class GRUCell:
    """This class represents a GRU unit"""
    def __init__(self, i, h, o):
        """Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """

        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))


        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))


        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """This method calculates the forward propagation for one time step
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data input
                 for the cell
        Returns: h_next, y
                 h_next: the next hidden state
                 y: the output of the cell
        """
        # Concatenate the previous hidden state (h_prev) and the input

        inputs = np.concatenate((h_prev, x_t), axis=1)

        update = self.sigmoid(np.matmul(inputs, self.Wz) + self.bz)

        # The reset gate (r_t) determines how much of the previous
        # hidden state to forget
        reset = self.sigmoid(np.matmul(inputs, self.Wr) + self.br)

        # Update the inputs (h_prev) with the (rest) reset gate
        # and concatenate with the input data (x_t)
        # axis=1 to concatenate horizontally
        updated_input = np.concatenate((reset * h_prev, x_t), axis=1)

        # Compute the new hidden state of the cell
        # h_r is shape (m, h)
        h_r = np.tanh(np.matmul(updated_input, self.Wh) + self.bh)

        # Calculate the new hidden state of the cell
        # after factroing in the update gate
        # h_next is shape (m, h)
        h_next = update * h_r + (1 - update) * h_prev

        # Calculate the output of the cell
        # taking in account the new hidden state
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, x):
        """This method calculates the sigmoid function
        Args:
            x: numpy.ndarray
        Returns: the sigmoid function of x
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """This method calculates the softmax function
        Args:
            x: numpy.ndarray
        Returns: the softmax function of x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)