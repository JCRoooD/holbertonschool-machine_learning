#!/usr/bin/env python3
"""based on 16-deep_neural_network.py"""
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        """Class constructor"""
        # Validate input types and values
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        # Initialize attributes
        self.__L = len(layers)  # Number of layers
        self.__cache = {}
        self.__weights = {}

        # Initialize weights using He et al. method
        for i in range(self.L):
            key = "W" + str(i + 1)
            if i == 0:
                # First layer weights based on nx
                self.weights[key] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                # Subsequent layers' weights based on previous layer
                self.weights[key] = \
                    np.random.randn(layers[i], layers[i - 1]) * np.sqrt(
                    2 / layers[i - 1]
                )
            # Initialize biases
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """getter method for number of layers"""
        return self.__L

    @property
    def cache(self):
        """getter method for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter method for weights"""
        return self.__weights
