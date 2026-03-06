"""
CORRECTED optimizers.py

CRITICAL CHANGE: REMOVED all weight clipping!
"""

import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0, **kwargs):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            grad_w = layer.grad_w + self.weight_decay * layer.W
            grad_b = layer.grad_b 
            
            layer.W -= self.lr * grad_w
            layer.b -= self.lr * grad_b
            
class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W  = None
        self.v_b  = None

    def _init_state(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            grad_w = layer.grad_w + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * grad_w
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * grad_b
            
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]

class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W  = None
        self.v_b  = None

    def _init_state(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            grad_w = layer.grad_w + self.weight_decay * layer.W
            
            v_W_prev = self.v_W[i].copy()
            v_b_prev = self.v_b[i].copy()
            
            self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * grad_w
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * layer.grad_b
            
            # Nesterov update logic
            layer.W -= self.lr * ((1 + self.beta) * self.v_W[i] - self.beta * v_W_prev)
            layer.b -= self.lr * ((1 + self.beta) * self.v_b[i] - self.beta * v_b_prev)

class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0, **kwargs):
        self.lr = learning_rate
        self.beta = beta
        self.eps  = epsilon
        self.weight_decay = weight_decay
        self.s_W  = None
        self.s_b  = None

    def _init_state(self, layers):
        if self.s_W is None:
            self.s_W = [np.zeros_like(l.W) for l in layers]
            self.s_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            grad_w = layer.grad_w + self.weight_decay * layer.W
            grad_b = layer.grad_b
            
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (grad_w ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (grad_b ** 2)
            
            layer.W -= self.lr * grad_w / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr * grad_b / (np.sqrt(self.s_b[i]) + self.eps)


OPTIMIZERS = {
    'sgd': SGD,
    'momentum': Momentum,
    'nag': NAG,
    'rmsprop': RMSProp,
}