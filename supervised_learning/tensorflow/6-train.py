#!/usr/bin/env python3
"""This module contains the function train"""
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

calculate_accuracy = __import__("3-calculate_accuracy").calculate_accuracy
calculate_loss = __import__("4-calculate_loss").calculate_loss
create_placeholders = __import__("0-create_placeholders").create_placeholders
create_train_op = __import__("5-create_train_op").create_train_op
forward_prop = __import__("2-forward_prop").forward_prop


def train(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    layer_sizes,
    activations,
    alpha,
    iterations,
    save_path="/tmp/model.ckpt",
):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
    X_train: numpy.ndarray containing the training input data.
    Y_train: numpy.ndarray containing the training labels.
    X_valid: numpy.ndarray containing the validation input data.
    Y_valid: numpy.ndarray containing the validation labels.
    layer_sizes: List containing the number of nodes in each
    layer of the network.
    activations: List containing the activation functions for
    each layer of the network.
    alpha: The learning rate.
    iterations: The number of iterations to train over.
    save_path: Designates where to save the model.

    Returns:
    The path where the model was saved.
    """
    # Create placeholders for the input data and labels
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Create the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate the accuracy
    accuracy = calculate_accuracy(y, y_pred)

    # Calculate the loss
    loss = calculate_loss(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Add to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    # Create a saver to save the model
    saver = tf.train.Saver()

    # Start a session to train the network
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            # Calculate the cost and accuracy for the training data
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train}
            )

            # Calculate the cost and accuracy for the validation data
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

    return save_path
