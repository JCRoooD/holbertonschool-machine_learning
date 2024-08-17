#!/usr/bin/env python3
import numpy as np


class GaussianProcess:
    """ Gaussian Process """
    
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Class constructor """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Calculates the covariance kernel matrix between two matrices """
        # sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        a = np.sum(X1 ** 2, axis=1, keepdims=True)
        b = np.sum(X2 ** 2, axis=1, keepdims=True)
        c = np.matmul(X1, X2.T)

        dist_sq = a + b.reshape(1, -1) - 2 * c

        K = (self.sigma_f ** 2) * np.exp(-0.5 * (1 / (self.l ** 2)) * dist_sq)
        return K

    def predict(self, X_s):
        """ Predicts the mean and standard deviation of points in a Gaussian Process """
        # Compute the kernel matrix
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        return mu, sigma

    def update(self, X_new, Y_new):
        """ Updates a Gaussian Process """
        # Update X and Y
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
        return None
