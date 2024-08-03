#!/usr/bin/env python3
""" Module initialize cluster centroids for K-means """
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means
        X: np.ndarray (n, d) dataset
          n: number of data points
          d: number of dimensions
        k: positive int, number of clusters
        Returns: np.ndarray (k, d) of initialized centroids
    """
    # Check if X is a 2D numpy array
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Get the number of data points (n) and dimensions (d)
    n, d = X.shape

    # Check if k is a valid integer within the range
    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    # Initialize centroids randomly within the range of the dataset
    centroids = np.random.uniform(
        low=np.min(X, axis=0), high=np.max(X, axis=0), size=(k, d)
    )
    return centroids
