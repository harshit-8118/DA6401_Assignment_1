"""
Optimizers for MLP training.
Supported: SGD, Momentum, NAG, RMSProp
"""
import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, **kwargs):
        self.lr = learning_rate

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b
        
        for layer in layers:
            np.clip(layer.W, -10.0, 10.0, out=layer.W)
            np.clip(layer.b, -10.0, 10.0, out=layer.b)


class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9, **kwargs):
        self.lr   = learning_rate
        self.beta = beta
        self.v_W  = None
        self.v_b  = None

    def _init_state(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W    -= self.lr * self.v_W[i]
            layer.b    -= self.lr * self.v_b[i]

        for layer in layers:
            np.clip(layer.W, -10.0, 10.0, out=layer.W)
            np.clip(layer.b, -10.0, 10.0, out=layer.b)

class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9, **kwargs):
        self.lr   = learning_rate
        self.beta = beta
        self.v_W  = None
        self.v_b  = None

    def _init_state(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def lookahead(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            layer.W -= self.beta * self.v_W[i]
            layer.b -= self.beta * self.v_b[i]

    def restore(self, layers):
        for i, layer in enumerate(layers):
            layer.W += self.beta * self.v_W[i]
            layer.b += self.beta * self.v_b[i]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W    -= self.lr * self.v_W[i]
            layer.b    -= self.lr * self.v_b[i]

        for layer in layers:
            np.clip(layer.W, -10.0, 10.0, out=layer.W)
            np.clip(layer.b, -10.0, 10.0, out=layer.b)

class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, **kwargs):
        self.lr   = learning_rate
        self.beta = beta
        self.eps  = epsilon
        self.s_W  = None
        self.s_b  = None

    def _init_state(self, layers):
        if self.s_W is None:
            self.s_W = [np.zeros_like(l.W) for l in layers]
            self.s_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * layer.grad_W ** 2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * layer.grad_b ** 2
            layer.W    -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b    -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)

        for layer in layers:
            np.clip(layer.W, -10.0, 10.0, out=layer.W)
            np.clip(layer.b, -10.0, 10.0, out=layer.b)


OPTIMIZERS = {
    'sgd':      SGD,
    'momentum': Momentum,
    'nag':      NAG,
    'rmsprop':  RMSProp,
}