#!/usr/bin/env python
# coding=utf-8
import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters:
    z : A scalar or numpy array of any size.

    Returns:
    s : sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def cost_function(X: np.ndarray, y: np.ndarray, theta, is_regularized=False, lambda_=0):
    """
    Compute the cost for logistic regression.

    Parameters:
    X : numpy array of shape (m, n), where m is the number of training examples and n is the number of features.
    y : numpy array of shape (m,), containing the labels (0 or 1).
    theta : numpy array of shape (n,), containing the parameters for logistic regression.
    is_regularized : boolean, whether to use regularization or not.
    lambda_ : float, regularization parameter.

    Returns:
    J : float, the computed cost.
    """
    m = len(y)  # number of training examples
    h = sigmoid(X.dot(theta))  # hypothesis function

    # Compute the cost
    J = (-1 / m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))

    # Add regularization term if needed
    if is_regularized:
        J += (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))

    return J
