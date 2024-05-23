#!/usr/bin/env python3
""" Confusion Matrix """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix """
    return np.dot(labels.T, logits)
