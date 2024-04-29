#!/usr/bin/env python3
""" Decision Tree """
import numpy as np


class Node:
    """
    Class that represents a node in the decision tree
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes a Node instance
        Args:
            feature: The feature used for splitting the node
            threshold: The value used for splitting the node
            left_child: The left child of the node
            right_child: The right child of the node
            is_root: Indicates if the node is the root
            depth: The depth of the node in the tree
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
        Calculates the maximum depth below the current node
        Returns:
            The maximum depth below the current node
        """
        if not self.left_child and not self.right_child:
            return self.depth

        left_depth = self.depth
        right_depth = self.depth

        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()

        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below the current node
        Args:
            only_leaves: If True, only counts leaf nodes
        Returns:
            The number of nodes below the current node
        """
        if only_leaves and not self.is_leaf:
            return (self.left_child.count_nodes_below(only_leaves=True)
                    if self.left_child else 0) + \
                (self.right_child.count_nodes_below(only_leaves=True)
                 if self.right_child else 0)

        count = 1 if not only_leaves or self.is_leaf else 0

        count += self.left_child.count_nodes_below(
            only_leaves) if self.left_child else 0
        count += self.right_child.count_nodes_below(
            only_leaves) if self.right_child else 0

        return count

    def __str__(self):
        """
        Returns the string representation of the current node
        """
        node_str = (
            f"root [feature={self.feature}, threshold={self.threshold}]\n"
            if self.is_root else
            f"-> node [feature={self.feature}, "
            f"threshold={self.threshold}]\n"
        )

        if self.is_leaf:
            return node_str

        left_str = self.left_child_add_prefix(
            self.left_child.__str__()) if self.left_child else ""
        right_str = self.right_child_add_prefix(
            self.right_child.__str__()) if self.right_child else ""

        return node_str + left_str + right_str

    def left_child_add_prefix(self, text):
        """ Adds a prefix to the string representation of the left child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """ Adds a prefix to the string representation of the right child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def get_leaves_below(self):
        """ Returns the leaves below the current node """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            leaves += self.left_child.get_leaves_below()
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Update the bounds of the leaves below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), self.threshold)
                else:  # right child
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()


class Leaf(Node):
    """
    Class that represents a leaf node in the decision tree
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf instance
        Args:
            value: The value of the leaf node
            depth: The depth of the leaf node in the tree
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf node
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 as a leaf node is a single node
        """
        return 1

    def __str__(self):
        """
        Returns the string representation of the leaf node
        """
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """ Returns the leaf node """
        return [self]

    def update_bounds_below(self):
        """ Does nothing as leaf nodes do not have children """
        pass


class Decision_Tree():
    """
    Class that represents a decision tree
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a Decision_Tree instance
        Args:
            max_depth: The maximum depth of the tree
            min_pop: The minimum population size for a node to be split
            seed: The seed for the random number generator
            split_criterion: The criterion used for splitting nodes
            root: The root node of the tree
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
        Returns the depth of the decision tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree
        Args:
            only_leaves: If True, only counts leaf nodes
        Returns:
            The number of nodes in the decision tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns the string representation of the decision tree
        """
        return self.root.__str__()

    def get_leaves(self):
        """ Returns the leaves of the decision tree """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Updates the bounds of the decision tree """
        self.root.update_bounds_below()
