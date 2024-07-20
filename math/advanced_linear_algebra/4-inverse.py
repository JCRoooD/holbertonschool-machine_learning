#!/usr/bin/env python3
"""" 4. Inverse matrix"""
adjugate = __import__('3-adjugate').adjugate
determinant = __import__('5-determinant').determinant


def inverse(matrix):
    """Calculates the inverse of a matrix
        Args:
        matrix: A list of lists whose inverse should be calculated
        Returns: The inverse of the
        matrix or None if the matrix is singular
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    # Validate the matrix is square and non-empty
    height = len(matrix)
    width = len(matrix[0])

    if height != width or (height == 1 and width == 0):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    detA = determinant(matrix)
    # Chck is its a singular matrix
    if detA == 0:
        return None

    adjugateZ = adjugate(matrix)
    # Iterate over the rowns and columns
    # of the adjugate matrix
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            # Divide each element of the adjugate matrix
            # by the determinant of the matrix
            adjugateZ[row][col] /= detA
    return adjugateZ
