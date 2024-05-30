#!/usr/bin/env python3
""" test model function"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ tests a neural network"""
    return network.evaluate(data, labels, verbose=verbose)
