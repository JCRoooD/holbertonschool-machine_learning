#!/usr/bin/env python3
"""marginal module"""
import numpy as np


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data"""
    if len(P) != len(Pr):
        raise ValueError("P and Pr must have the same length")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood of the data for each probability in P
    likelihood = (P ** x) * ((1 - P) ** (n - x))

    # Calculate the joint probability (intersection) of the data and the hypothesis
    intersection = likelihood * Pr

    # Calculate the marginal probability using the law of total probability
    marginal_prob = np.sum(intersection)

    return marginal_prob
