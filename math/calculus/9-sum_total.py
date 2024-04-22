#!/usr/bin/env python3
"""Sigma function"""


def summation_i_squared(n):
    """function that sums in sigma notation"""
    if not isinstance(n, int) and n < 0:
        return None

    squared_sum = int(n * (n + 1) * (2*n+1) / 6)

    return squared_sum
