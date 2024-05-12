#!/usr/bin/env python3
"""evaluate"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        feed_dict = {x: X, y: Y}
        accuracy = sess.run(accuracy, feed_dict)
        loss = sess.run(loss, feed_dict)
        return accuracy, loss
