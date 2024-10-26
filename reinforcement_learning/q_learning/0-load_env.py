#!/usr/bin/env python3
""" this module contains the function load_frozen_lake """
import gymnasium as gym


def load_frozen_lake(desc=None, map_name='4x4', is_slippery=False):
    """
    loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
    desc: is either None or a list of lists containing a custom description
    map_name: is a string containing the pre-made map to load
    is_slippery: is a boolean to determine if the ice is slippery
    Returns: the environment
    """
    env = gym.make(
        'FrozenLake-v1', desc=desc, map_name=map_name,
                is_slippery=is_slippery, render_mode='ansi')
    return env
