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
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        D = cache['D' + str(i)]
        if i == L:
            dW = np.matmul(dZ, A.T) / m
        else:
            dZ = np.matmul(W.T, dZ) * (1 - A ** 2) * D
            dZ /= keep_prob
            dW = np.matmul(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db