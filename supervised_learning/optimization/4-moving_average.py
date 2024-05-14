#!/usr/bin/env python3
"""moving average"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    v = 0
    moving_averages = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
