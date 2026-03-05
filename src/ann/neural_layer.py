import numpy as np
from .activations import ACTIVATIONS


def initialize_weights(weight_init, input_size, output_size):
    if weight_init == 'zeros':
        return np.zeros((input_size, output_size))
    elif weight_init == 'random':
        return np.random.randn(input_size, output_size) * 0.01
    else:  # xavier
        std = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * std


class NeuralLayer:
    def __init__(self, input_size, output_size, activation='relu',
                 weight_init='xavier', layer_name='hidden'):

        self.W = initialize_weights(weight_init, input_size, output_size)
        self.b = np.zeros((1, output_size))

        self.X  = None   # input  (batch, input_size)
        self.Z  = None   # pre-activation
        self.A  = None   # post-activation

        self.layer_name = layer_name

        self.activation,      self.activation_grad = ACTIVATIONS[activation]
        self.activation_name = activation
        if layer_name == 'output':
            self.activation_grad = None

        self.grad_b = None
        self.grad_w = None   

        #  Dead-neuron / activation statistics 
        self.dead_neuron_counts = []   # fraction of neurons dead per batch
        self.activation_history = []  # mean activation per neuron (for plotting)
        self.grad_history       = []
        
    #  Forward 
    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self.activation(self.Z)

        # Track dead neurons (only for ReLU hidden layers)
        if self.activation_name == 'relu' and self.layer_name == 'hidden':
            dead_fraction = np.mean(self.A == 0)
            self.dead_neuron_counts.append(float(dead_fraction))
            # Mean activation magnitude per neuron across the batch
            self.activation_history.append(self.A.mean(axis=0).copy())

        return self.A

    #  Backward 
    def backward(self, delta, weight_decay=0.0):
        if self.layer_name != 'output':
            dz = delta * self.activation_grad(self.Z)
        else:
            dz = delta   

        # Compute gradients
        self.grad_w = self.X.T @ dz + weight_decay * self.W  # Shape: (input_size, output_size)
        self.grad_b = np.sum(dz, axis=0, keepdims=True)      # Shape: (1, output_size)
        
        # if self.layer_name == 'hidden':
        #     self.grad_history.append(np.abs(dz).mean(axis=0))
        
        return dz @ self.W.T

    #  Utilities 
    def dead_neuron_fraction(self, X_probe):
        Z = X_probe @ self.W + self.b
        A = self.activation(Z)
        # A neuron is dead if it fires 0 for ALL samples
        dead = np.all(A == 0, axis=0)
        return float(np.mean(dead)), dead

    def activation_distribution(self, X_probe):
        Z = X_probe @ self.W + self.b
        A = self.activation(Z)
        return {
            'mean'        : A.mean(axis=0),
            'std'         : A.std(axis=0),
            'percent_zero': (A == 0).mean(axis=0) * 100,
            'min'         : A.min(axis=0),
            'max'         : A.max(axis=0),
            'raw'         : A,
        }

    def gradient_flow_summary(self):
        if not self.grad_history:
            return None
        return np.stack(self.grad_history, axis=0).mean(axis=0)

    def _is_relu_hidden(self):
        return self.activation_name == 'relu' and self.layer_name == 'hidden'