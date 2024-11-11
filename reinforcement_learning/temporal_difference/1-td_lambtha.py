#!/usr/bin/env python3
""" Temporal Difference task 1. TD(λ) """
import numpy as np


def td_lambtha(env, V, policy, lambtha=1, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """ Function that performs the TD(λ) algorithm 
        on a reinforcement learning task """
    for episode in range(episodes):
        # reset the environment and get initial state
        state = env.reset()[0]

        # Init. eligibility traces to zero, for all states
        eligibility_traces = np.zeros_like(V)

        for step in range(max_steps):
            # Select action based on policy
            action = policy(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD Error (δ): reward + gamma * V(next_state) - V(state)
            delta = reward + (gamma * V[next_state] - V[state])

            # Update eligibility trace for the current state
            eligibility_traces[state] += 1

            # Update each state's value and eligibility trace
            V += alpha * delta * eligibility_traces

            # Apply lambtha decay to eligibility traces
            eligibility_traces *= gamma * lambtha

            # Move to the next state
            state = next_state

            if terminated or truncated:
                break

    return V

