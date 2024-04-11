#!/usr/bin/env python3
"""Module that returns the transpose of a matrix
    """


def matrix_transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
