#!/usr/bin/env python3
"""normalization constants"""
import numpy as np


def normalization_constants(X):
    """normalization constants"""
    return np.mean(X, axis=0), np.std(X, axis=0)
