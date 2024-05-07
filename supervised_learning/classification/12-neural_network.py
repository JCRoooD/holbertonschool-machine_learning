#!/usr/bin/env python3
"""based on 11-neural_network.py"""
import numpy as np


class NeuralNetwork:
    """Initializes a neural network with
    one hidden layer to binary classification"""
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
        """Getter to weights"""
        return self.__W1

    @property
    def b1(self):
        """Getter to bias"""
        return self.__b1

    @property
    def A1(self):
        """Getter to activated output"""
        return self.__A1

    @property
    def W2(self):
        """Getter to weights"""
        return self.__W2

    @property
    def b2(self):
        """Getter to bias"""
        return self.__b2

    @property
    def A2(self):
        """Getter to activated output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the toward propagation of the neural network"""
        # Z1 is the dot product of weights and
        #           input data plus bias to the hidden layer
        Z1 = np.matmul(self.W1, X) + self.b1
        # Apply sigmoid activation function
        #           to Z1 to get A1 (output of hidden layer)
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Z2 is the dot product of weights
        # and A1 plus bias to the output layer
        Z2 = np.matmul(self.W2, self.__A1) + self.b2
        # Apply sigmoid activation function
        #           to Z2 to get A2 (final output)
        self.__A2 = 1 / (1 + np.exp(-Z2))

        # Return A1 and A2
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        # Get the number of examples
        m = Y.shape[1]

        # Compute the cost using the frmula fr logistic regression
        # The cost is the average of the losses calculated fr each example
        # The loss fr each example is calculated using the frmula:
        # -(y * log(a) + (1 - y) * log(1 - a))
        # A small value (1.0000001) is subtracted
        #           from 1 to avoid division by zero errors
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m

        # Return the cost
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        # Generate the prediction
        A1, A2 = self.forward_prop(X)

        # Calculate the cost
        cost = self.cost(Y, A2)

        # Generate the prediction
        prediction = np.where(A2 >= 0.5, 1, 0)

        # Return the prediction and cost
        return prediction, cost
