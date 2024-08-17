#!/usr/bin/env python3
""" Bayesian Optimization - Acquisition """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """This class performs Bayesian optimization on a
    noiseless 1D Gaussian process
    """

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True,
    ):
        """This method initializes the class BayesianOpt
        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
            t: the number of initial samples
            bounds: tuple of (min, max) representing the bounds of the space
                    in which to look for the optimal point
            ac_samples: the number of samples that should be analyzed during
                        acquisition
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                    black-box function
            xsi: the exploration-exploitation factor for acquisition
            minimize: a bool determining whether optimization should be
                    performed for minimization (True) or maximization (False)
            return: None
        """
        self.f = f
        # gp is an instance of the GaussianProcess class
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], num=ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """This method calculates the next best sample location
        Returns: X_next, EI
        X_next: numpy.ndarray of shape (1,) representing the next best sample
        EI: numpy.ndarray of shape (ac_samples,) containing the expected
            improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_sample_opt = np.min(self.gp.Y)
            imp = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            imp = mu - Y_sample_opt - self.xsi

        A = imp / sigma

        EI = imp * norm.cdf(A) + sigma * norm.pdf(A)

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """ This method optimizes the black-box function
        Args:
            iterations: the maximum number of iterations to perform
            Returns: X_opt, Y_opt
        """
        all_X = []

        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in all_X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            all_X.append(X_next)

        if self.minimize is True:
            Y_opt = np.min(self.gp.Y)
            idx = np.argmin(self.gp.Y)
        else:
            Y_opt = np.max(self.gp.Y)
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]

        return X_opt, Y_opt
