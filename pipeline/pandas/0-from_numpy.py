#!/usr/bin/env python3
""" From Numpy module"""
import pandas as pd


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray"""
    columns = [chr(i) for i in range(65, 91)]
    return pd.DataFrame(array, columns=columns[:array.shape[1]])
