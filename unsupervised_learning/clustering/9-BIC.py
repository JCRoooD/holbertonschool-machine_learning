#!/usr/bin/env python3
""" module contains BIC function """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """This function finds the best number of clusters fr a GMM using the
    Bayesian Information Criterion
    """
    # Step 1: veryify inputs

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None

    # Step 2: Extract the shape from X
    n, d = X.shape

    # Step 3: Initialize the variables
    likelihoods = []
    pis = []
    ms = []
    Ss = []
    bys = []

    # Step 4: Perform the EM algorithm fr each cluster size
    for i in range(kmin, kmax + 1):
        pi, m, S, g, likelihood = expectation_maximization(
            X, i, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        pis.append(pi)
        ms.append(m)
        Ss.append(S)
        likelihoods.append(likelihood)

        # Calculate the BIC
        p = (i * d * (d + 1) / 2) + (d * i) + (i - 1)
        bic = p * np.log(n) - 2 * likelihood
        bys.append(bic)

    # Step 5: Find the best number of clusters
    likelihoods = np.array(likelihoods)
    bys = np.array(bys)
    best_k = np.argmin(bys)
    best_result = (pis[best_k], ms[best_k], Ss[best_k])

    return best_k+1, best_result, likelihoods, bys
