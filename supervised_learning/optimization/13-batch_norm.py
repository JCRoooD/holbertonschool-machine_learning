#!/usr/bin/env python3
"""
normalizes an unactivated output of a neural
network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output (Z) of a neural network using batch 
    normalization.

    Parameters:
    Z (numpy.ndarray): The unactivated output of a neural network. 
    gamma (numpy.ndarray): A numpy array used for
    element-wise scaling of the normalized Z.
    beta (numpy.ndarray): A numpy array used for
    element-wise shifting of the normalized Z.
    epsilon (float): A small number added for numerical stability.

    Returns:
    Z_tilda (numpy.ndarray): The batch-normalized output.

    Batch normalization is a technique for improving the speed, performance, 
    and stability of artificial neural networks. It is used to normalize the 
    input layer by adjusting and scaling the activations.
    """
    # Calculate the mean of Z
    m = Z.shape[0]
    mean = np.sum(Z, axis=0) / m

    # Calculate the variance of Z
    variance = np.sum((Z - mean) ** 2, axis=0) / m

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift the normalized Z
    Z_tilda = gamma * Z_norm + beta

    return Z_tilda
