#!/usr/bin/env python3
"""train"""

import tensorflow.compat.v1 as tf


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"
):
    """Builds, trains, and saves a neural network classifier"""
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_train_op = __import__('5-create_train_op').create_train_op

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_train = sess.run(y_pred, feed_dict={x: X_train, y: Y_train})
            acc_valid = sess.run(y_pred, feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        return tf.train.Saver().save(sess, save_path)
