#!/usr/bin/env python3
"""Defining class Neuron"""
import numpy as np


class Neuron:
    """Defining a single neuron"""
    def __init__(self, nx):
        """constructor class initializing the instance Neuron"""
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0
