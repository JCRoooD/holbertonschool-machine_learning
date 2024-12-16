#!/usr/bin/env python3
""" hue image function"""
import tensorflow as tf


def change_hue(image, delta):
    """ randomly changes the hue of an image"""
    return tf.image.adjust_hue(image, delta)
