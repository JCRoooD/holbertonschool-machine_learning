#!/usr/bin/env python3
""" variance module """
import numpy as np


def variance(X, C):
    """ calculates the total intra-cluster variance for a data set
        X: np.ndarray (n, d) dataset
          n: number of data points
          d: number of dimensions
        C: np.ndarray (k, d) centroid means for each cluster
          k: number of clusters
        Returns: var, or None on failure
          var: total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    # Step 1: Extract the shape of the dataset
    n, d = X.shape

    # Step 2: Extract the number of clusters
    k = C.shape[0]

    # Check if K is valid
    if k > n:
        return None
    # Check the that the shape of the centroids is correct
    if d != C.shape[1]:
        return None

    # Step 3: vectorize the data sets
    vectorized_data_X = np.repeat(X, k, axis=0)
    # reshape the data set
    # n: the shap of the first dimention of the new shape
    # k: the shape of the second dimention of the new shape
    # d: the shape of the third dimention of the new shape
    vectorized_data_X = vectorized_data_X.reshape(n, k, d)

    # Step 4: vectorize the centroids
    vectorized_centroids = np.tile(C, (n, 1))
    # Reshape the centroids to (n, k, d)
    # n: number of data points
    # k: number of clusters
    # d: number of dimensions
    vectorized_centroids = vectorized_centroids.reshape(n, k, d)

    # Step 5: Calculate the squared Euclidean distance between each data point and each centroid
    distance = np.linalg.norm(vectorized_data_X - vectorized_centroids, axis=2)

    # Step 6: Determine the minimum squared distance for each data point
    # Square each element of the distance
    dist_short = np.min(distance ** 2, axis=1)

    # Sum up the minimum squared distances to get the total variance
    variance = np.sum(dist_short)

    return variance
