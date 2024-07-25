#!/usr/bin/env python3
"""intersection module """
import numpy as np


def intersection(x, n, P, Pr):
    """ intersection - calculates the intersection of obtaining this data with
    the various hypothetical probabilities
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
        a 1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((0 <= P) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((0 <= Pr) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood using the binomial probability formula
    likelihood = (np.math.factorial(n) / 
                  (np.math.factorial(x) * np.math.factorial(n - x))) * (P ** x) * ((1 - P) ** (n - x))
    
    # Calculate the intersection
    intersection = likelihood * Pr
    
    return intersection
