#!/usr/bin/env python3
"""This modlue initializes variables fr a Gaussian
Mixture Model
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """This function initializes variables fr a Gaussian Mixture Model
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
    Returns:
    pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
    m: numpy.ndarray of shape (k, d) containing the centroid means fr
    each cluster
    S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
    fr each cluster
    or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None

    pi = np.full((k,), 1 / k)

    m, _ = kmeans(X, k)

    S = np.full((k, d, d), np.identity(d))

    return pi, m, S
