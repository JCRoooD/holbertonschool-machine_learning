#!/usr/bin/env python3
import numpy as np

def posterior(x, n, P, Pr):
    """Calculates the posterior probability of developing severe side effects"""
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood of the data for each probability in P
    likelihood = (P ** x) * ((1 - P) ** (n - x))

    # Calculate the joint probability (intersection) of the data and the hypothesis
    intersection = likelihood * Pr

    # Calculate the marginal probability using the law of total probability
    marginal_prob = np.sum(intersection)

    # Calculate the posterior probability
    posterior_prob = intersection / marginal_prob

    return posterior_prob
