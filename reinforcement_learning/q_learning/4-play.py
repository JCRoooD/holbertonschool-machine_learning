#!/usr/bin/env python3
""" agent that plays an episode of FrozenLake """
import numpy as np


def play(env, Q, max_steps=100):
    """ plays an episode of FrozenLake
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode
        Returns: the total rewards for the episode
    """
    state = env.reset()
    done = False
    total_rewards = 0
    rendered_outputs = []
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())
        state = new_state
        if done:
            break
    return total_rewards, rendered_outputs
