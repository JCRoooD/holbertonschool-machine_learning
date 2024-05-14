#!/usr/bin/env python3
"""normalize a matrix"""
import numpy as np


def normalize(X, m, s):
    """normalize a matrix"""
    return (X - m) / s
