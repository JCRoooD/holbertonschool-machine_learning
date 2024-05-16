#!/usr/bin/env python3
"""update learning rate using inverse time decay with tensorflow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Update the learning rate using inverse time decay with tensorflow.

    Parameters:
    alpha (float): Original learning rate.
    decay_rate (float): Decay rate.
    decay_step (int): Number of passes of gradient descent needed to occur
    before the learning rate is decayed further.

    Returns:
    Updated learning rate.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
