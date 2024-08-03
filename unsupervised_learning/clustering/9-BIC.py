#!/usr/bin/env python3
""" module contains BIC function """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a GMM using the Bayesian Information Criterion:
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: int, minimum number of clusters to check for (inclusive)
        kmax: int, maximum number of clusters to check for (inclusive)
        iterations: int, maximum number of iterations for the EM algorithm
        tol: float, tolerance for the EM algorithm
        verbose: bool, determines if the EM algorithm should print information to the standard output
    
    Returns:
        best_k: int, best value for k based on its BIC
        best_result: tuple, containing pi, m, S
        l: numpy.ndarray of shape (kmax - kmin + 1), log likelihood for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1), BIC value for each cluster size tested
    """
    if kmax is None:
        kmax = X.shape[0]

    n, d = X.shape
    l = []
    b = []
    best_k = None
    best_result = None
    best_bic = float('inf')

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or log_likelihood is None:
            return None, None, None, None

        l.append(log_likelihood)

        # Calculate the number of parameters
        p = k * d + k * d * (d + 1) / 2 + k - 1
        bic = p * np.log(n) - 2 * log_likelihood
        b.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(l), np.array(b)
