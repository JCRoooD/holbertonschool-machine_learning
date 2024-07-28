#!/usr/bin/env python3
""" Mean and Covariance """
import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set:"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0).reshape(1, d)
    cov = np.dot((X - mean).T, (X - mean)) / (n - 1)
    return mean, cov
