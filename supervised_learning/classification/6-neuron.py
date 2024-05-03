#!/usr/bin/env python3
"""Based on 5-neuron.py"""
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
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions and the cost of the network.

        Arguments:
        X: numpy.ndarray with shape (nx, m) that contains the input data.
        Y: numpy.ndarray with shape (1, m) that
        contains the correct labels for the input data.

        Returns the neuron’s prediction and the
        cost of the network, respectively.
        """
        # Get the activated output of the neuron
        A = self.forward_prop(X)
        # Calculate the cost
        cost = self.cost(Y, A)
        # Get the predictions. If the output of the
        # network is >= 0.5, the label is 1. Otherwise, it's 0.
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Arguments:
        X: numpy.ndarray with shape (nx, m) that contains the input data.
        Y: numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data.
        A: numpy.ndarray with shape (1, m) containing the
            activated output of the neuron for each example.
        alpha: learning rate

        Updates the private attributes __W and __b.
        """
        m = X.shape[1]
        # Calculate the gradient of the loss with respect to A
        dZ = A - Y
        # Calculate the gradient of the loss with respect to W
        dW = np.matmul(dZ, X.T) / m
        # Calculate the gradient of the loss with respect to b
        db = np.sum(dZ) / m
        # Update W and b
        self.__W -= alpha * dW
        self.__b -= alpha * db

    """
    This function first calculates the gradients of the loss with respect
    to A, W, and b. The gradient of the loss with respect to A is A - Y.
    The gradient of the loss with respect to W is the dot product of dZ and
    the transpose of X, divided by m. The gradient of the loss with respect
    to b is the sum of dZ, divided by m. The function then updates W and b by
    subtracting alpha times dW and db from W and b, respectively.
    """

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
    Trains the neuron by updating the private attributes __W, __b, and __A.
    """
        # Validate the input parameters
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            # Perform forward propagation and calculate the cost
            self.forward_prop(X)
            # Perform backpropagation and update the weights and bias
            self.gradient_descent(X, Y, self.__A, alpha)

        # Evaluate the training data after all iterations have occurred
        return self.evaluate(X, Y)
