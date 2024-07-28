#!/usr/bin/env python3
""" calculates correlation matrix """
import numpy as np


def correlation(C):
    """ calculates the correlation matrix of a covariance matrix """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d = np.diag(C)
    d = np.sqrt(d)
    d = np.outer(d, d)
    Co = C / d
    return Co
