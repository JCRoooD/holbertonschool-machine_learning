#!/usr/bin/env python3
"""based on 23-deep_neural_network.py"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], nx
                ) * np.sqrt(2 / nx)
            else:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])
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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        # Store input data in cache
        self.__cache["A0"] = X
        # Loop over each layer
        for i in range(self.L):
            # Generate keys to weights, biases, and activations
            keyW = "W" + str(i + 1)
            keyb = "b" + str(i + 1)
            keyA = "A" + str(i)
            key_newA = "A" + str(i + 1)
            # Retrieve weights, biases, and previous activations
            W = self.weights[keyW]
            A = self.cache[keyA]
            b = self.weights[keyb]
            # Compute weighted sum
            Z = np.matmul(W, A) + b
            # Apply sigmoid activation function
            self.__cache[key_newA] = 1 / (1 + np.exp(-Z))
        # Return final activation and cache
        return self.cache[key_newA], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        # Number of examples
        m = Y.shape[1]
        # Compute the cost
        calc = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(calc)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        # Generate predictions
        A, _ = self.forward_prop(X)
        # Calculate the predicted labels
        Y_hat = np.where(A >= 0.5, 1, 0)
        # Calculate the cost
        cost = self.cost(Y, A)
        # Return the predicted labels and cost
        return Y_hat, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            cache: dictionary containing all intermediary values of the network
            alpha: learning rate
        """
        # Number of examples in input data
        m = Y.shape[1]
        # Calculate the gradients of the output data
        dZ = cache["A" + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            # Get the cached activations
            A_prev = cache["A" + str(i - 1)]
            # Calculate the derivatives of the weights and biases
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            # Calculate the derivative of the cost
            # with respect to the activation
            dZ_step1 = np.dot(self.weights["W" + str(i)].T, dZ)
            dZ = dZ_step1 * (A_prev * (1 - A_prev))
            # Update the weights and biases
            self.weights["W" + str(i)] -= alpha * dW
            self.weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        # Validate the number of iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        # Validate the learning rate
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize variables
        cost_list = []

        for i in range(iterations + 1):
            # Perform forward propagation
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    cost_list.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), cost_list)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as file:
                obj = pickle.load(file)
            return obj
        except FileNotFoundError:
            return None
        except Exception:
            return None
