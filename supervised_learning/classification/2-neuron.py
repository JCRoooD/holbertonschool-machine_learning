#!/usr/bin/env python3
"""Module to define a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Constructor for the neuron class
        Arguments:
        nx: number of input features to the neuron
        """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights vector for the neuron
        self.__W = np.random.randn(nx).reshape(1, nx)
        # Bias for the neuron
        self.__b = 0
        # Output of the neuron
        self.__A = 0

    @property
    def W(self):
        """Getter to the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter to the bias"""
        return self.__b

    @property
    def A(self):
        """Getter to the output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Arguments:
        X: numpy.ndarray with shape (nx, m) that contains the input data
        """
        # Calculate the net input
        w_sum = np.matmul(self.__W, X) + self.__b
        # Calculate the neuron's output
        self.__A = 1 / (1 + np.exp(-w_sum))
        return self.__A
