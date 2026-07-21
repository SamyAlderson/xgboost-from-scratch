# src/utils.py

import numpy as np
from scipy.special import expit

def sigmoid(x):
    """
    Compute the sigmoid of x.

    This is used as the link function in logistic regression.
    """
    return 1 / (1 + np.exp(-x))

def log_loss(y, y_pred):
    """
    Compute the log loss between y and y_pred.

    y is true labels (0 or 1)
    y_pred is predicted probabilities (0 to 1)
    """
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def gradient(x, y):
    """
    Compute the gradient of the log loss with respect to x.

    This is used in gradient descent optimization.
    """
    return 2 * (y - sigmoid(x))

def hessian(x):
    """
    Compute the Hessian matrix of the log loss with respect to x.

    This is used in Newton's method optimization.
    """
    return np.diag([sigmoid(x) * (1 - sigmoid(x)) for _ in range(len(x))])

def calculate_gain(left_gain, right_gain, split_point, feature):
    """
    Compute the gain of a split.

    This is used in decision tree construction.
    """
    if left_gain == right_gain:
        return np.inf  # cannot split if gain is 0

    left_size = np.sum(feature < split_point)
    right_size = len(feature) - left_size

    gain = left_gain * left_size + right_gain * right_size
    return gain