#!/usr/bin/env python3
""" Dropout Gradient Descent """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights of a neural network with Dropout regularization
        Y: one-hot numpy.ndarray of shape (classes, m) that contains the
            correct labels for the data
            classes: number of classes
            m: number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
        All layers use the tanh activation function except the last
        The last layer uses the softmax activation function
        The weights of the network should be updated in place
    """
    m = Y.shape[1]
    A_prev = cache['A' + str(L - 1)]
    A_curr = cache['A' + str(L)]
    dZ = A_curr - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dA_prev = np.dot(W.T, dZ)
            D = cache['D' + str(layer - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)  # Derivative of tanh

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
