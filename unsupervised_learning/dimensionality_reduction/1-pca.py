#!/usr/bin/env python3
""" PCA dataset """
import numpy as np


def pca(X, ndim):
    """ performs PCA on a dataset
        X: np.ndarray (n, d) dataset
          n: number of data points
          d: number of dimensions
        ndim: new dimensionality of the transformed X
        Returns: T, a np.ndarray of shape (n, ndim) with the transformed X
    """
    X = X - np.mean(X, axis=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(X)

    # Compute the weights matrix
    W = Vt[:ndim].T

    # Transform the data
    T = np.matmul(X, W)

    return T
