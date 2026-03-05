"""
Loss / Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

# MEAN SQUARED ERROR 
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    batch_size = y_true.shape[0] * y_true.shape[1]
    return (2.0 / batch_size) * (y_pred - y_true)

# CROSS-ENTROPY
def cross_entropy(y_true, y_pred):
    eps   = 1e-15
    probs = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(probs)) /  y_pred.shape[0]

def cross_entropy_derivative(y_true, y_pred):
    epsilon    = 1e-9
    batch_size = y_true.shape[0]
    y_pred     = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / batch_size


OBJECTIVE = {
    'mse':           (mse,           mse_derivative),
    'cross_entropy': (cross_entropy, cross_entropy_derivative),
}