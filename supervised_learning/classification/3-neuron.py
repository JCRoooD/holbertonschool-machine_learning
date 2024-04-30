#!/usr/bin/env python3
"""Based on 2-neuron.py"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Constructor to the neuron class
        Arguments:
        nx: number of input features to the neuron
        """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights vector to the neuron
        self.__W = np.random.randn(nx).reshape(1, nx)
        # Bias to the neuron
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
        net_sum = np.matmul(self.__W, X) + self.__b
        # Calculate the neuron's output
        self.__A = 1 / (1 + np.exp(-net_sum))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Arguments:
        Y: numpy.ndarray with shape (1, m) that contains
        the correct labels to the input data.

        A: numpy.ndarray with shape (1, m) containing
        the activated output of the neuron to each example.

        The function calculates the cost using
        the formula to binary cross-entropy loss.
        """
        m = Y.shape[1]  # Number of examples
        # Binary cross-entropy loss is a loss function
        #                       used to binary classification problems.
        # It measures the dissimilarity between the true
        #                       label (Y) and the predicted probability (A).
        # The loss is high when the model predicts a
        #                       probability close to 0 to a positive instance,
        #                       and when it predicts a probability close
        #                       to 1 to a negative instance.
        # The '1.0000001 - A' term is used to avoid division by zero errors.
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
