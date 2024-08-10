#!/usr/bin/env python3
""" regular function markov chain """
import numpy as np


def regular(P):
    """ function that determines the steady state probabilities of a regular
        markov chain
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.any(P <= 0):
        return None

    n = P.shape[0]
    s = np.ones((1, n)) / n
    while True:
        s_prev = s
        s = np.matmul(s, P)
        if np.all(s == s_prev):
            return s
