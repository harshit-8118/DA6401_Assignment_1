"""
Loss / Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

# MEAN SQUARED ERROR 
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    batch_size = y_true.shape[0]
    return 2.0 * (y_pred - y_true) / batch_size

# CROSS-ENTROPY
def cross_entropy(y_true, y_pred):
    eps = 1e-15
    probs = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(probs)) /  y_pred.shape[0]

def cross_entropy_derivative(y_true, y_pred_logits):
    z_shifted = y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    batch_size = y_pred_logits.shape[0]
    return (probs - y_true) / batch_size


OBJECTIVE = {
    'mse': (mse, mse_derivative),
    'cross_entropy': (cross_entropy, cross_entropy_derivative),
}