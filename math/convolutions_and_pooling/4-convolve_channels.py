#!/usr/bin/env python3
"""Convultion with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """convultion of images through channels
    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        containing multiple images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the image
        kernel: numpy.ndarray with shape (kh, kw, c)
        containing the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding: tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
        stride: tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
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
    output = np.zeros((m, output_h, output_w))
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                image_padded[:, i * sh: i * sh + kh,
                             j * sw: j * sw + kw, :] * kernel,
                axis=(1, 2, 3),
            )
    return output
