#!/usr/bin/env python3
""" Specificity """
import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in a confusion matrix """
    total = np.sum(confusion)
    # We can get the True Positives (TP) by summing the diagonal
    TP = np.diag(confusion)
    # Sum each row (actual positives)
    actual_positives = np.sum(confusion, axis=1)
    # Sum each column (predicted positives)
    predicted_positives = np.sum(confusion, axis=0)
    # True Negatives (TN) are all values that are not in the
    # actual or predicted
    TN = total - (actual_positives + predicted_positives - TP)
    # False Positives (FP) are the values that are not in the actual
    # but in the predicted
    FP = predicted_positives - TP
    # Calculate the specificity for each class
    return TN / (TN + FP)
