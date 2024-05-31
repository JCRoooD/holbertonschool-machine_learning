#!/usr/bin/env python3
"""Multiple Kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ that performs a convolution on images using multiple kernels:"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh - h + kh) / 2) + 1
        pw = int(((w - 1) * sw - w + kw) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    output_h = int((h - kh + (2 * ph)) / sh) + 1
    output_w = int((w - kw + (2 * pw)) / sw) + 1
    output = np.zeros((m, output_h, output_w, nc))
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant')

    kernels = kernels.reshape((1, *kernels.shape))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                image_padded[:, i * sh: i * sh + kh,
                              j * sw: j * sw + kw, :, None] * kernels,
                axis=(1, 2, 3),
            )
    return output
