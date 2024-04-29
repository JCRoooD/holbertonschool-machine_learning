#!/usr/bin/env python3
""" Decision Tree Implementation """
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
        Initialize a node with given feature, threshold,
        left and right children, root status, and depth.
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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below the current node in the decision tree.
        """
        # ct = count
        ct = 0
        if not only_leaves or self.is_leaf:
            ct = 1
        if not self.is_leaf:
            ct += self.left_child.count_nodes_below(only_leaves=only_leaves)
            ct += self.right_child.count_nodes_below(only_leaves=only_leaves)

        return ct

    def __str__(self):
        """
        Method that returns the string representation of the current node
        """
        # String representation for the current node
        node_str = (
            f"root [feature={self.feature}, threshold={self.threshold}]\n"
            if self.is_root
            else f"-> node [feature={self.feature}, " f"threshold={self.threshold}]\n"
        )

        # If the node is a leaf, simply return the string representation
        if self.is_leaf:
            return node_str

        # Formatting for the left and right children
        left_str = (
            self.left_child_add_prefix(self.left_child.__str__())
            if self.left_child
            else ""
        )
        right_str = (
            self.right_child_add_prefix(self.right_child.__str__())
            if self.right_child
            else ""
        )

        return node_str + left_str + right_str

    def left_child_add_prefix(self, text):
            """ Add prefix to the left child """
            lines = text.split("\n")
            # Adding prefix to the first line
            new_text = "    +--" + lines[0] + "\n"
            # Adding prefix to the rest of the lines
            new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
            # Append an additional newline character if there are multiple lines
            new_text += "\n" if len(lines) > 1 else ""
            return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child """
        lines = text.split("\n")
        # Adding prefix to the first line
        new_text = "    +--" + lines[0] + "\n"
        # Adding prefix to the rest of the lines
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        # Append an additional newline character if there are multiple lines
        new_text += "\n" if len(lines) > 1 else ""
        return new_text


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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below the current node in the decision tree.
        """
        return 1


def __str__(self):
    return f"-> leaf [value={self.value}] "


class Decision_Tree:
    """
    Class representing a decision tree.
    """

    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None
    ):
        """
        Initialize a decision tree with given maximum depth,
        minimum population, seed, split criterion, and root.
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

    def count_nodes(self, only_leaves=False):
        """ct nodes in decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)


def __str__(self):
    return self.root.__str__()
