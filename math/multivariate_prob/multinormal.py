#!/usr/bin/env python3
""" MultiNormal class"""
import numpy as np


class MultiNormal:
    """MultiNormal class"""

    def __init__(self, data):
        """Constructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (data.shape[1] - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != self.mean.shape[0]:
            raise ValueError("x must have the shape ({}, 1)".format(self.mean.shape[0]))
        n = self.mean.shape[0]
        x_m = x - self.mean
        pdf = 1 / np.sqrt(((2 * np.pi) ** n) * np.linalg.det(self.cov)) * np.exp(
            -0.5 * np.dot(np.dot(x_m.T, np.linalg.inv(self.cov)), x_m))
        return pdf.item()
