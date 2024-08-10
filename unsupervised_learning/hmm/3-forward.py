#!/usr/bin/env python3
""" forward algorithm """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model
        Observation: np arr (T,) of index of observation
                     T: number of observations
        Emission: np arr (N, M) of emission probabilities
                   N: number of hidden states
                   M: number of possible observations
        Transition: np arr (N, N) of transition probabilities
                     Transition[i, j]: prob of transitioning from i to j
        Initial: np arr (N, 1) of prob of starting in each state
        Returns: P, F
                 P: likelihood of the observations given the model
                 F: np arr (N, T) of forward path probabilities
                    F[i, j]: prob of being in state i at time j given the obs
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            F[j, i] = np.sum(F[:, i - 1] * Transition[:, j] *
                             Emission[j, Observation[i]])
    P = np.sum(F[:, -1])
    return P, F
