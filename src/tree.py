# tree.py: Implementation of the decision tree algorithm used by XGBoost

import numpy as np

class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """Implementation of the decision tree algorithm."""
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Builds the decision tree from the training data."""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Recursively builds the decision tree."""
        if depth >= self.max_depth or X.shape[0] == 1:
            return Node(value=np.mean(y))

        feature_indices = np.random.choice(X.shape[1], 2, replace=False)
        feature_values = X[:, feature_indices]

        thresholds = np.unique(np.concatenate((feature_values, [X.min(), X.max()])))
        thresholds = thresholds[1:-1]

        best_feature, best_threshold = self._find_best_split(X, y, feature_indices, thresholds)

        if best_feature is None:
            return Node(value=np.mean(y))

        left_indices = X[:, best_feature] < best_threshold
        right_indices = np.logical_not(left_indices)

        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _find_best_split(self, X, y, feature_indices, thresholds):
        """Finds the best split for the decision tree."""
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')

        for threshold in thresholds:
            left_indices = X[:, feature_indices[0]] < threshold
            right_indices = np.logical_not(left_indices)

            left_gain, right_gain = self._calculate_gain(X, y, left_indices, right_indices)

            if left_gain + right_gain > best_gain:
                best_gain = left_gain + right_gain
                best_feature = feature_indices[0]
                best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gain(self, X, y, left_indices, right_indices):
        """Calculates the gain for the decision tree."""
        left_y = y[left_indices]
        right_y = y[right_indices]

        left_entropy = self._calculate_entropy(left_y)
        right_entropy = self._calculate_entropy(right_y)

        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)

        left_weight = num_left / (num_left + num_right)
        right_weight = num_right / (num_left + num_right)

        gain = np.sum(left_entropy) * left_weight + np.sum(right_entropy) * right_weight

        return gain

    def _calculate_entropy(self, y):
        """Calculates the entropy for the decision tree."""
        unique_values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def predict(self, X):
        """Makes predictions on the input data."""
        predictions = []
        for row in X:
            node = self.root
            while node.value is None:
                if row[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return np.array(predictions)