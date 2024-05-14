#!/usr/bin/env python3
"""Module for calculating the moving average of a data set"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Parameters:
    data (list or numpy array): The data set for
    which the moving average is to be calculated.
    beta (float): The weight used for the moving average, where 0 <= beta < 1.
                  A higher beta discounts older observations faster.

    Returns:
    moving_averages (list): The moving averages for
    each data point in the data set.
    """

    v = 0  # Initialize the moving average
    moving_averages = []  # Initialize the list to store the moving averages

    # Loop over each data point
    for i in range(len(data)):
        # Update the moving average using the formula:
        # v_t = beta * v_(t-1) + (1 - beta) * x_t
        v = beta * v + (1 - beta) * data[i]

        # Correct the bias in the initial data points
        # Bias correction helps adjust for the fact
        # that the initial moving averages have been
        # calculated from fewer data points
        unbiased_v = v / (1 - beta ** (i + 1))

        # Append the unbiased moving average to the list
        moving_averages.append(unbiased_v)

    # Return the list of moving averages
    return moving_averages
