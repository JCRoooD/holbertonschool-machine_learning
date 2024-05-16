#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using RMSProp optimization algorithm.

    Parameters:
    alpha (float): Learning rate, determines step size at each iteration.
    beta2 (float): RMSProp weight, used for moving
    average of squared gradient.
    epsilon (float): Small number to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of var.
    s (numpy.ndarray): Previous second moment of var.

    Returns:
    var_update (numpy.ndarray): Updated variable.
    s (numpy.ndarray): New moment.
    """

    # Calculate moving average of squared gradients
    s = beta2 * s + (1 - beta2) * grad ** 2

    # Update the variable
    var_update = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var_update, s
