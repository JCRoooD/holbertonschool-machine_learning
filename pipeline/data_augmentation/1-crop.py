#!/usr/bin/env python3
""" crop image function"""
import tensorflow as tf


def crop_image(image, size):
    """ crops an image to a specified size"""
    return tf.image.random_crop(image, size)
