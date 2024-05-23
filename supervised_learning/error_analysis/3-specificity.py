#!/usr/bin/env python3
""" Specificity """
import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in a confusion matrix """
    return np.diag(confusion) / np.sum(confusion, axis=0)
