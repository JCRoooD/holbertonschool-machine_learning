#!/usr/bin/env python3
""" Dropout Forward Propagation """
import numpy as np


def tanh_active(X):
    """ tanh activation function """
    return np.tanh(X)

def softmax_active(X):
    """ softmax activation function """
    return np.exp(X) / np.sum(np.exp(X), axis=0)


def dropout_forward_prop(X, weights, L, keep_prob):
    """ conducts forward propagation using Dropout """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        # calc linear combination of the inputs
        W_key = "W" + str(i)
        A_key = "A" + str(i - 1)
        b_key = "b" + str(i)
        Z = np.matmul(weights[W_key], cache[A_key]) + weights[b_key]
        if i != L:
            # Apply tanh activation function and dropout
            A = tanh_active(X)
            # Convert boolean to int
            random_array = np.random.rand(A.shape[0], A.shape[1])
            D = (random_array < keep_prob).astype(int)
            A = np.multiply(A, D)
            A /= keep_prob
            cache["D" + str(i)] = D
        else:
            # Apply softmax activation function
            A = softmax_active(X)
        cache["A" + str(i)] = A

    return cache
