#!/usr/bin/env python3
"""marginal module"""
import numpy as np


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != 1:
        raise TypeError("Pr must be a 1D numpy.ndarray")
    if P.shape != Pr.shape:
        raise ValueError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood of the data for each probability in P
    factorial = np.math.factorial
    likelihood = (factorial(n) / (factorial(x) * factorial(n - x))) * (P ** x) * ((1 - P) ** (n - x))

    # Calculate the marginal probability
    marginal_prob = np.sum(likelihood * Pr)

    return marginal_prob
