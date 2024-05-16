#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    This function updates a variable using the RMSProp optimization algorithm.
    
    Parameters:
    alpha (float): The learning rate. This is a hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function.
    beta2 (float): The RMSProp weight. This is a hyperparameter that is used for the moving average of the squared gradient.
    epsilon (float): A small number to avoid division by zero. This is used to maintain numerical stability.
    var (numpy.ndarray): A numpy array containing the variable to be updated. This is the variable that the optimization algorithm will update to minimize the loss function.
    grad (numpy.ndarray): A numpy array containing the gradient of var. This is the vector of partial derivatives of the loss function with respect to var.
    s (numpy.ndarray): The previous second moment of var. This is an exponentially weighted infinity norm. It stores an exponentially decaying average of past squared gradients.
    
    Returns:
    var_update (numpy.ndarray): The updated variable after applying the RMSProp optimization algorithm.
    s (numpy.ndarray): The new moment. This is the updated value of s, which is used for the next update.
    """
    
    # Calculate the moving average of the squared gradients
    s = beta2 * s + (1 - beta2) * grad ** 2
    
    # Update the variable
    var_update = var - alpha * grad / (np.sqrt(s) + epsilon)
    
    return var_update, s
