#!/usr/bin/env python3
"""This module contains the YOLO class"""
import tensorflow.keras as K


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
        pass
