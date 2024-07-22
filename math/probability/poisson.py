#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Poisson distribution"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha value
            self.lambtha = float(sum(data) / len(data))
        else:
            # If data is not given
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        Calculates the value of the PMF for a given number of “successes”
        Args:
            k (int): The number of “successes”
            Returns:
        """
        e = 2.7182818285

        if not isinstance(k, int):
            k = int(k)

        # The PMF is 0 for negative values of k, as negative
        # successes are not possible.
        if k < 0:
            return 0

        # Calculate k factorial (k!) as it's required for the
        # PMF formula.
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate and return the PMF using the formula
        # P(k; lambda) = (e^-lambda) * (lambda^k) / k!
        # This formula represents the probability of
        # observing exactly k successes.
        return ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial
