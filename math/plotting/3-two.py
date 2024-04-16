#!/usr/bin/env python3
""" 2. Two is better than one """
import numpy as np
import matplotlib.pyplot as plt


def two():
    """Exponentional Decay of Radioactive Elements"""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, color="red", linestyle="--")
    plt.plot(x, y2, color="green")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.legend(["C-14", "Ra-226"])
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.show()