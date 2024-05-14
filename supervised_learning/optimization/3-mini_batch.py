#!/usr/bin/env python3
"""mini batch"""
import numpy as np


def create_mini_batches(X, Y, batch_size):
    """creates mini-batches"""
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    complete_batches = m // batch_size

    for i in range(0, complete_batches):
        X_batch = X[i * batch_size: i * batch_size + batch_size, :]
        Y_batch = Y[i * batch_size: i * batch_size + batch_size, :]
        mini_batches.append((X_batch, Y_batch))

    if k * batch_size < m:
        X_batch = X[complete_batches * batch_size: m, :]
        Y_batch = Y[complete_batches * batch_size: m, :]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
