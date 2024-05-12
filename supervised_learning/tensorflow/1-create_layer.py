#!/usr/bin/env python3
"""This module defines a function  create a neuron layer
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
Creates a neural network layer with weights, biases, and an activation function.
"""
# Initialize weights with He et. al method
initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

# Define weights and biases
weights = tf.Variable(initializer(shape=(int(prev.shape[1]), n)), name='weights')
biases = tf.Variable(tf.zeros([n]), name='biases')

# Compute weighted sum of inputs and apply activation function
layer = tf.add(tf.matmul(prev, weights), biases)
