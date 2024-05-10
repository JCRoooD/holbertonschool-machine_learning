#!/usr/bin/env python3
"""one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if len(Y) == 0:
        return None
    if classes <= np.max(Y):
        return None
    try:
        matrix = np.zeros((classes, len(Y)))

        matrix[Y, np.arange(len(Y))] = 1
        return matrix

    except Exception:
        return None
