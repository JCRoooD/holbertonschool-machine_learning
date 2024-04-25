#!/usr/bin/env python3

import numpy as np


class Node:
    """
    Class representing a node in the decision tree.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):
        """
        Initialize a node with given feature, threshold, left and right children, root status, and depth.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Return the maximum depth below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(), self.right_child.max_depth_below()
            )


class Leaf(Node):
    """
    Class representing a leaf in the decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf with given value and depth.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf.
        """
        return self.depth


class Decision_Tree:
    """
    Class representing a decision tree.
    """

    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None
    ):
        """
        Initialize a decision tree with given maximum depth, minimum population, seed, split criterion, and root.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Return the depth of the root of the decision tree.
        """
        return self.root.max_depth_below()
