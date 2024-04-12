#!/usr/bin/env python3
"""Module that performs element-wise addition, subtraction,
multiplication, and division"""


def np_elementwise(mat1, mat2):
    """Performs element-wise actions on two matrices"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
