#!/usr/bin/env python3
""" Early Stopping """
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if you should stop gradient descent early
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used to determine early stopping
        patience: patience count used for early stopping
        count: how long the threshold has not been met
        Returns: boolean of whether the network should be stopped early,
                 followed by the updated count
    """
    if cost > opt_cost + threshold:
        count += 1
    else:
        count = 0

    if count == patience:
        return True, count
    return False, count
