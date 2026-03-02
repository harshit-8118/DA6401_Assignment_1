"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


#  TANH 
def tanh(x):
    return np.tanh(x)   

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

#  SIGMOID 
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):    
    s = sigmoid(z)
    return s * (1 - s)

#  RELU 
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)


#  SOFTMAX 
def softmax(z):
    z_stable = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def softmax_derivative(z):
    raise NotImplementedError(
        "softmax_derivative should never be called directly. "
        "Use the combined CE+softmax gradient: (probs - y) / batch_size"
    )

identity = lambda z: z
identity_grad = lambda z: np.ones_like(z)

ACTIVATIONS = {
    'tanh': (tanh, tanh_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'identity': (identity, identity_grad),
    'softmax': (softmax, softmax_derivative)
}

