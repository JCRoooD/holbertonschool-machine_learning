#!/usr/bin/env python3
""" K-means module """
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


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset
        X: np.ndarray (n, d) dataset
          n: number of data points
          d: number of dimensions
        k: positive int, number of clusters
        iterations: positive int, number of iterations
        Returns: np.ndarray (k, d) of centroids, np.ndarray (n,) of indices
    """
    centroids = initialize(X, k)

    if centroids is None:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    for iteration in range(iterations):
        # Keep a copy of the prev centroids
        prev_centroids = np.copy(centroids)

        X_vector = np.repeat(X, k, axis=0)

        X_vector = X_vector.reshape(n, k, d)

        centroids_vector = np.tile(centroids, (n, 1))

        centroids_vector = centroids_vector.reshape(n, k, d)

        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)

        clss = np.argmin(distance ** 2, axis=1)

        for i in range(k):

            indixes = np.where(clss == i)[0]
            if len(indixes) == 0:
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = np.mean(X[indixes], axis=0)

        if np.all(prev_centroids == centroids):
            return centroids, clss

        centroids_vector = np.tile(centroids, (n, 1))
        centroids_vector = centroids_vector.reshape(n, k, d)
        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)
        clss = np.argmin(distance ** 2, axis=1)

    return centroids, clss
