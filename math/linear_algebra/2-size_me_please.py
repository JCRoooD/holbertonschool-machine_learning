#!/usr/bin/env python3
"""Module that return the shape of a matrix
    """


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    Args:
        matrix (list): The matrix to calculate the shape of.

    Returns:
        list: A list representing the shape of
        the matrix. Each element in the list
              corresponds to the size of a dimension in the matrix.

    Example:
        >>> matrix = [[1, 2, 3], [4, 5, 6]]
        >>> matrix_shape(matrix)
        [2, 3]
    """
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
