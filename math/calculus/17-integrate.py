#!/usr/bin/env python3
"""poly integral"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int) or not poly:
        return None

    integral = [C]

    for i in range(len(poly)):
        if isinstance(poly[i], (int, float)):
            result = poly[i] / (i + 1)
            # Check if the result is a whole number
            if result.is_integer():
                # Convert to integer
                result = int(result)
            integral.append(result)
        else:
            return None

    while integral[-1] == 0 and len(integral) > 1:
        integral.pop()

    return integral
