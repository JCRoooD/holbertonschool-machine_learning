#!/usr/bin/env python3
"""This module contains a BidirectionaCell class of an RNN"""
import numpy as np


class BidirectionalCell:
    """This class represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """

        self.Whf = np.random.normal(size=(i + h, h))

        self.bhf = np.zeros((1, h))

        self.Whb = np.random.normal(size=(i + h, h))

        self.bhb = np.zeros((1, h))

        self.Wy = np.random.normal(size=(2 * h, o))

        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """This method calculates the forward propagation for one time step
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
                    m: is the batch size
                    h: is the dimensionality of the hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data input
                 for the cell
                 m: is the batch size
                 i: is the dimensionality of the data
        Returns: h_next
                 h_next: the next hidden state
        """

        cell_input = np.concatenate((h_prev, x_t), axis=1)
        # Print the shape of the cell input
        # print(f"cell input shape: {cell_input.shape}")

        # Calculate the forward hidden state
        # The shape of the forward hidden state is (m, h)
        h_next = np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)
        # Print the shape of the forward hidden state
        # print(f"forward hidden state shape: {h_next.shape}")

        return h_next
