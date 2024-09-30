#!/usr/bin/env python3
"""module contains the function for positional encoding
    in a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
        """
    # Create a matrix of shape (max_position, d_model) with zeros
    angle_rads = np.zeros((max_seq_len, dm))

    # Compute the angles for each position and dimension
    for pos in range(max_seq_len):
        for i in range(dm):
            angle_rads[pos, i] = pos / np.power(
                10000, (2 * (i // 2)) / dm)

    # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
