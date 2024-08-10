#!/usr/bin/env python3
""" markov chain module """
import numpy as np


def markov_chain(P, s, t=1):
    """ function that determines the probability of a markov chain being in
        a particular state after a specified number of iterations
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if P.shape[0] != P.shape[1] or s.shape[1] != P.shape[0]:
        return None

    for _ in range(t):
        s = np.matmul(s, P)

    return s
