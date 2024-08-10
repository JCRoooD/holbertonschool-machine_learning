#!/usr/bin/env python3
""" backward algorithm """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ performs backward algorithm for hidden markov model
        Observation: np arr (T,) of index of observation
                     T: number of observations
        Emission: np arr (N, M) of emission probabilities
                   N: number of hidden states
                   M: number of possible observations
        Transition: np arr (N, N) of transition probabilities
                     Transition[i, j]: prob of transitioning from i to j
        Initial: np arr (N, 1) of prob of starting in each state
        Returns: P, B
                 P: likelihood of the observations given the model
                 B: np arr (N, T) of backward path probabilities
                    B[i, j]: prob of generating the future observations from
                             hidden state i at time j
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
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for i in range(T - 2, -1, -1):
        for j in range(N):
            B[j, i] = np.sum(B[:, i + 1] * Transition[j, :] *
                             Emission[:, Observation[i + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
