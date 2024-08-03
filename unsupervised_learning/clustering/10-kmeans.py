#!/usr/bin/env python3
""" module to perform k means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int, number of clusters
    
    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
        clss: numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
