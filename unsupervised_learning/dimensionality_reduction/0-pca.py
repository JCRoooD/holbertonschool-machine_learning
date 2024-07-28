#!/usr/bin/env python3
""" PCA module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a specified fraction of variance.

    X: np.ndarray of shape (n, d)
       n is the number of data points
       d is the number of dimensions in each point
       All dimensions have a mean of 0 across all data points.
    var: float
         Fraction of the variance that the PCA transformation should maintain.
```
    Returns:
    W: np.ndarray of shape (d, nd)
       Weights matrix that maintains var fraction of X's original variance.
       nd is the new dimensionality of the transformed X.
    """
    # Perform SVD
    U, S, Vt = np.linalg.svd(X)

    # Compute the cumulative sum of the explained variance ratio
    sum_s = np.cumsum(S)

    # Infer 'r' (number of principal components to extract from W/V)
    # based on the 'var' treshold passed as argument to the method
    # Normalize sum_s:
    sum_s = sum_s / sum_s[-1]

    r = np.min(np.where(sum_s >= var))

    # Compute Vr(= Wr):
    V = Vt.T
    Vr = V[..., :r + 1]

    return Vr
