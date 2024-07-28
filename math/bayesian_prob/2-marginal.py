#!/usr/bin/env python3
"""marginal module"""
import numpy as np


def likelihood(x, n, P):
    """this method calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
        Returns: a 1D numpy.ndarray containing the likelihood of obtaining the
        data, x and n, for each probability in P, respectively
        """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    # Check if x is a integer thats grater or equal to 0
    if not isinstance(x, int) or x < 0:
        error = "x must be an integer that is greater than or equal to 0"
        raise ValueError(error)
    # Check if x i9s grater than n
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Product of probabilities for the data:
    A = (P ** x) * ((1 - P) ** (n - x))
    # Factorials to be accounted for:
    B = np.math.factorial(x) * np.math.factorial(n - x) / np.math.factorial(n)
    L = A / B

    return L


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
    if not isinstance(x, int) or x <= 0:
        error = "x must be an integer that is greater than or equal to 0"
        raise ValueError(error)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Likelihood of obtaining data x and n with each probability in P
    LH = likelihood(x, n, P)
    # Intersection of obtaining x and n with each probability in P
    intersection = LH * Pr

    return intersection


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
