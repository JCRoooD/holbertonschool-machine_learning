#!/usr/bin/env python3
"""Module that multiplies two matrices"""


def mat_mul(mat1, mat2):
    """Multiplication of matrices"""
    if len(mat1[0]) != len(mat2):
        return None

    # Nested list comprehension performs matrix multiplication.
    # Outer loop iterates over each row in the first matrix.
    # Inner loop iterates over each column in the second matrix.
    # `zip(*mat2)` transposes the second matrix.
    # `sum(a * b for a, b in zip(mat1_row, mat2_col))` multiplies and sums up.
    # The sum is an element in the resulting matrix.
    return [
        [
            sum(a * b for a, b in zip(mat1_row, mat2_col))
            for mat2_col in zip(*mat2)
        ]
        for mat1_row in mat1
    ]
