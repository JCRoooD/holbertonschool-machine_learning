#!/usr/bin/env python3
"""Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable using the Adam optimization algorithm.

    Parameters:
    alpha (float): Learning rate, determines step size at each iteration.
    beta1 (float): Weight used for the first moment.
    beta2 (float): Weight used for the second moment.
    epsilon (float): Small number to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of var.
    v (numpy.ndarray): Previous first moment of var.
    s (numpy.ndarray): Previous second moment of var.
    t (int): Time step used to calculate bias-corrected moments.

    Returns:
    var_update (numpy.ndarray): Updated variable.
    v (numpy.ndarray): New first moment.
    s (numpy.ndarray): New second moment.
    """
    # Calculate moving averages of gradients
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2

    # Bias correction
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Update the variable
    var_update = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_update, v, s
