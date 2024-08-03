#!/usr/bin/env python3
""" module to perform agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering with Ward linkage and displays the dendrogram.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: float, maximum cophenetic distance for all clusters
    
    Returns:
        clss: numpy.ndarray of shape (n,) containing the cluster indices for each data point
    """
    # Perform agglomerative clustering using Ward linkage
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    
    # Create the dendrogram
    plt.figure()
    dendro = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    
    # Display the dendrogram
    plt.show()
    
    # Get the cluster indices for each data point
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    
    return clss
