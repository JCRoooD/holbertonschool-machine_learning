#!/usr/bin/env python3
"""convultion with padding"""
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h - kh + (2 * ph) + 1
    output_w = w - kw + (2 * pw) + 1
    output = np.zeros((m, output_h, output_w))
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(image_padded[:, i:i + kh, j:j + kw] * kernel,
                                     axis=(1, 2))
    return output
