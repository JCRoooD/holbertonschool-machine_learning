#!/usr/bin/env python3
"""shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
