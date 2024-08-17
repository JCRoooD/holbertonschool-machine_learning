#!/usr/bin/env python3
"""Gaussian Process module"""
import numpy as np


class GaussianProcess:
    """ class for4 Gaussian Process """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init: numpy.ndarray of shape (t, 1) representing the inputs already
                sampled with the black-box function
        Y_init: numpy.ndarray of shape (t, 1) representing the outputs of the
                black-box function for each input in X_init
        t: the number of initial samples
        l: the length parameter for the kernel
        sigma_f: the standard deviation given to the output of the black-box
                function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """This method calculates the covariance kernel matrix between two
        matrices
        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)
        Returns: the covariance kernel matrix as a numpy.ndarray of shape
                (m, n)

        Also, the kernel should use the Radial Basis Function (RBF) Kernel
        """
        sqdist = np.sum(X1**2, 1).reshape(
            -1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)


        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
