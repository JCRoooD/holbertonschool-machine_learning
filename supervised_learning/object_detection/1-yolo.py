#!/usr/bin/env python3
"""This module contains the YOLO class"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.activations import sigmoid  # type: ignore


class Yolo:
    """YOLO class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor
        Args:
            model_path: is the path to where a Darknet Keras model is stored
            classes_path: is the path to where the list of class names used
                          for the Darknet model, listed in order of index,
                          can be found
            class_t: is a float representing the box score threshold for the
                     initial filtering step
            nms_t: is a float representing the IOU threshold for non-max
            suppression
            anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all of the anchor boxes:
                     outputs: is the number of outputs (predictions) made
                     by the Darknet model
                     anchor_boxes: is the number of anchor boxes used
                     for each prediction
                     2: [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs: is a list of numpy.ndarrays containing the predictions
                     from the Darknet model for a single image:
                     Each output will have the shape (grid_height, grid_width,
                     anchor_boxes, 4 + 1 + classes)
                     grid_height & grid_width: the height and width of the grid
                     used for the output
                     anchor_boxes: the number of anchor boxes used
                     4: (t_x, t_y, t_w, t_h)
                     1: box_confidence
                     classes: class probabilities for all classes
            image_size: is a numpy.ndarray containing the image’s original
                        size [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
                   anchor_boxes, 4) containing the processed boundary boxes
                   for each output, respectively:
                   4: (x1, y1, x2, y2)
                   (x1, y1, x2, y2) should represent the boundary box
                   relative to original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
                             grid_width, anchor_boxes, 1) containing the box
                             confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
                            grid_width, anchor_boxes, classes) containing the
                            box’s class probabilities for each output,
                            respectively
        """
        boxes = []
        # List to hold the confidence for each box in each output
        box_confidences = []
        # List to hold the class probabilities for each box in each output
        box_class_probs = []

        # Unpack the outputs
        for i, output in enumerate(outputs):
            # Ig nore the rest with _
            grid_height, grid_width, anchor_boxes, _ = output.shape
            # Extract the box parameters
            box = output[..., :4]
            # Extract the individual components
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            # Create 3D grid for the anchor boxes
            # Create a grid for the x coordinates
            c_x = np.arange(grid_width).reshape(1, grid_width)
            #  Repeat the x grid anchor_boxes times
            c_x = np.repeat(c_x, grid_height, axis=0)
            # Reshape to add the anchor boxes
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            # Create a grid for the y coordinates
            c_y = np.arange(grid_width).reshape(1, grid_width)
            # Repeat the y grid anchor_boxes times
            c_y = np.repeat(c_y, grid_height, axis=0).T
            # Reshape to add the anchor boxes
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            # Create a grid for the anchor boxes
            b_x = (sigmoid(t_x) + c_x) / grid_width
            b_y = (sigmoid(t_y) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            # top left corner
            x1 = (b_x - b_w / 2)
            y1 = (b_y - b_h / 2)

            # bottom right corner
            x2 = (b_x + b_w / 2)
            y2 = (b_y + b_h / 2)

            # box coordinate relative to the image size
            x1 = x1 * image_size[1]
            y1 = y1 * image_size[0]
            x2 = x2 * image_size[1]
            y2 = y2 * image_size[0]

            # Update boxes
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            # Append the box to the boxes list
            boxes.append(box)

            # Extract the box confidence and aply sigmoid
            box_confidence = output[..., 4:5]
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_confidences.append(box_confidence)

            # Extract the box class probabilities and aply sigmoid
            box_class_prob = output[..., 5:]
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
