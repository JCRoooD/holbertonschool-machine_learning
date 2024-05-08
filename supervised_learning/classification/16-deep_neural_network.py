#!/usr/bin/env python3
"""deep neural network class"""
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network
    performing binary classification
    """
    def __init__(self, nx, layers):
        """class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            # Initialize weights using He et al. method
            # If it's the first layer, the weights are based
            # on the number of input features nx
            if i == 0:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            # Fr subsequent layers, the weights are based on the number
            # of nodes in the previous layer
            else:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2/layers[i - 1])
            # Initialize biases to 0's fr each layer
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))