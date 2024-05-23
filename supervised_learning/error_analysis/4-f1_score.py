#!/usr/bin/env python3
""" F1 Score """
import numpy as np


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix """
    precision = np.diag(confusion) / np.sum(confusion, axis=0)
    recall = np.diag(confusion) / np.sum(confusion, axis=1)
    return 2 * (precision * recall) / (precision + recall)
