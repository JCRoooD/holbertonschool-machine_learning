#!/usr/bin/env python3
""" module to perform gmm on a dataset"""
import sklearn.mixture


def gmm(X, k):
    """ calculates GMM on a dataset 
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int, number of clusters
    
    Returns:
        pi: numpy.ndarray of shape (k,) containing the cluster priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        clss: numpy.ndarray of shape (n,) containing the cluster indices for each data point
        bic: float, BIC value for the model
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)
    
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    
    return pi, m, S, clss, bic
