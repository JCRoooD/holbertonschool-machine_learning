#!/usr/bin/env python3
"""This module contains the function evaluate"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
    X: numpy.ndarray containing the input data to evaluate.
    Y: numpy.ndarray containing the one-hot labels for X.
    save_path: The location to load the model from.

    Returns:
    The network’s prediction, accuracy, and loss, respectively.
    """
    # Load the saved model
    saver = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        saver.restore(sess, save_path)

        # Retrieve the operations and tensors from the saved model
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Evaluate the network's prediction, accuracy, and loss
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})

    return prediction, accuracy, loss
