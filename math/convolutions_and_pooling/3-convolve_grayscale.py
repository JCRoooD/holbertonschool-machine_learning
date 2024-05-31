#!/usr/bin/env python3
"""Strided convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding="same", stride=(1, 1)):
    """performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == "same":
        ph = int(((h - 1) * sh - h + kh) / 2) + 1
        pw = int(((w - 1) * sw - w + kw) / 2) + 1
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        ph, pw = padding
    output_h = int((h - kh + (2 * ph)) / sh) + 1
    output_w = int((w - kw + (2 * pw)) / sw) + 1
    output = np.zeros((m, output_h, output_w))
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), "constant")
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                image_padded[:, i * sh : i * sh + kh, j * sw : j * sw + kw] * kernel,
                axis=(1, 2),
            )
    return output
