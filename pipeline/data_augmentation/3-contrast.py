#!/usr/bin/env python3
""" contrast image function"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """ changes the contrast of an image"""
    return tf.image.adjust_contrast(image, lower, upper)
