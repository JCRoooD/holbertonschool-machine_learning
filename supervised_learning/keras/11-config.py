#!/usr/bin/env python3
""" save and load config functions"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves an entire model's config"""
    with open(filename
                , 'w') as f:
            f.write(network.to_json())
    return None


def load_config(filename):
    """loads an entire model's config"""
    with open(filename
                , 'r') as f:
        return K.models.model_from_json(f.read())
