#!/usr/bin/env python3
"""based on 0-neuron.py"""
import numpy as np


class Neuron:
    """defining class Neuron"""

    def __init__(self, nx):
        """constructor class for instance of Neuron"""
        if isinstance(nx, int) is False:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter func"""
        return self.__W

    @property
    def b(self):
        """Getter func"""
        return self.__b

    @property
    def A(self):
        """getter func"""
        return self.__A
