#!/usr/bin/env python3
"""poly derivative"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    derivative = [poly[i] * i for i in range(1, len(poly))]

    if not derivative:
        return [0]

    return derivative
