#!/usr/bin/env python3
"""Class definition for binary
classification neural network"""
import numpy as np


class NeuralNetwork:
    """Initializes a neural network with
    one hidden layer for binary classification"""
    def __init__(self, nx, nodes):
        """constructor class"""
        # Validate input types and values
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize hidden layer weights, bias, and output
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize output neuron weights, bias, and output
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0


    @property
    def W1(self):
        """Getter for weights"""
        return self.__W1


    @property
    def b1(self):
        """Getter for bias"""
        return self.__b1


    @property
    def A1(self):
        """Getter for activated output"""
        return self.__A1


    @property
    def W2(self):
        """Getter for weights"""
        return self.__W2


    @property
    def b2(self):
        """Getter for bias"""
        return self.__b2


    @property
    def A2(self):
        """Getter for activated output"""
        return self.__A2
