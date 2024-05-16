#!/usr/bin/env python3
"""update learning rate using inverse time decay in numpy"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Update the learning rate using inverse time decay in numpy.

    Parameters:
    alpha (float): Original learning rate.
    decay_rate (float): Decay rate.
    global_step (int): Number of passes of gradient descent that have elapsed.
    decay_step (int): Number of passes of gradient descent needed to occur
    before the learning rate is decayed further.

    Returns:
    Updated learning rate.
    """
    alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))
    return alpha
