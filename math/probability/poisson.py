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
