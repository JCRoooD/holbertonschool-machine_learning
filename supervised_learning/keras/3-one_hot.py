#!/usr/bin/env python3
""" converting labels to one-hot matrix """
import numpy as np


def one_hot(labels, classes=None):
    """ converts a label vector into a one-hot matrix """
    if classes is None:
        classes = np.max(labels) + 1
    return np.eye(classes)[labels]
