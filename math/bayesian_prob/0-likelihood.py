#!/usr/bin/env python3
"""likelihood module"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
    :return: 1D numpy.ndarray containing the likelihood of obtaining the data, x and n, for each probability in P
    """
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be an integer that is greater than 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    fact = np.math.factorial
    comb = fact(n) / (fact(x) * fact(n - x))
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
