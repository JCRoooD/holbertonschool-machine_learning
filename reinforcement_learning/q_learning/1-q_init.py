#!/usr/bin/env python3
""" this module contains the function for task 1 """
import numpy as np


def q_init(env):
    """ initializes the Q-table:
        env: the FrozenLakeEnv instance
        Returns: the Q-table as a numpy.ndarray of zeros
    """
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    return q_table

