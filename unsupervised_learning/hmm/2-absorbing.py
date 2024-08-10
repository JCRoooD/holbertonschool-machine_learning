#!/usr/bin/env python3
""" absorbing module """
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    # Ensure the sum of all probabilities in each row is equal to 1
    num_states = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(num_states))[0]:
        return None

    # Check if the matrix has at least one absorbing state
    # An absorbing state is one where P[i, i] == 1
    if np.all(np.diag(P) != 1):
        return False

    # If all states are absorbing (P is the identity matrix)
    if np.all(np.diag(P) == 1):
        return True

    # Step 5: Check if every state can reach an absorbing state
    for i in range(num_states):
        if np.any(P[i, :] == 1):
            continue
        break

    sub_matrix_I = P[:i, :i]
    identity_matrix = np.identity(num_states - i)
    sub_matrix_R = P[i:, :i]
    sub_matrix_Q = P[i:, i:]

    # Step 6: Calculate the fundamental matrix
    try:
        fundamental_matrix = np.linalg.inv(identity_matrix - sub_matrix_Q)
    except Exception:
        return False

    # Step 7: Calculate the product of the fundamental matrix and sub_matrix_R
    FR_product = np.matmul(fundamental_matrix, sub_matrix_R)
    limiting_matrix = np.zeros((num_states, num_states))
    limiting_matrix[:i, :i] = sub_matrix_I
    limiting_matrix[i:, :i] = FR_product

    sub_matrix_Qbar = limiting_matrix[i:, i:]
    # Step 8: Check if the sub_matrix_Qbar is zero
    if np.all(sub_matrix_Qbar == 0):
        return True

    return False
