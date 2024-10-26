#!/usr/bin/env python3
""" this module contains the function for task 2 """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ uses epsilon-greedy to determine the next action:
        Q: numpy.ndarray containing the q-table
        state: current state
        epsilon: epsilon to use for the calculation
        Returns: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        explore = np.random.randint(Q.shape[1])
    else:
        explore = np.argmax(Q[state])
    
    return explore
