#!/usr/bin/env python3
""" module contains EM function """

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """  that performs the expectation maximization for a GMM: """
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Initialize variables
    l_prev = 0
    for i in range(iterations):
        # Expectation step
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Maximization step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Check fr convergence
        if abs(l - l_prev) <= tol:
            break
        l_prev = l

        # Log likelihood if verbose is True
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

    # Final log likelihood
    if verbose:
        print(f"Log Likelihood after {i + 1} iterations: {l:.5f}")

    return pi, m, S, g, l
