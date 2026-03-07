"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import json
import os
import argparse

import numpy as np

from ann.neural_layer import NeuralLayer
from ann.activations import ACTIVATIONS
from ann.objective_functions import OBJECTIVE
from ann.optimizers import OPTIMIZERS
from utils.data_loader import train_val_split, compute_metrics


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.weight_decay = getattr(cli_args, 'weight_decay', 0.0)

        self.loss, self.loss_grad = OBJECTIVE[getattr(cli_args, 'loss', 'cross_entropy')]
        self._best_val_f1 = -1.0
        self._best_epoch  = 0

        self.grad_history_layer0 = []

        self.hidden_size = getattr(cli_args, 'hidden_size', [128, 128, 64])
        
        # Re-initialize loss in case it was added above
        self.loss, self.loss_grad = OBJECTIVE[cli_args.loss]

        # Build layer sizes: [784] + hidden + [10]
        self.is_built = False 
        self.layers = []

        # Build optimizer
        opt_name  = cli_args.optimizer
        opt_cls   = OPTIMIZERS[opt_name]
        self.is_nag = (opt_name == 'nag')
        bet = getattr(cli_args, 'beta',    0.9)
        epsilon = getattr(cli_args, 'epsilon', 1e-8)

        if opt_name == 'sgd':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate)
        elif opt_name in ('momentum', 'nag'):
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta=self.cli_args.beta)
        elif opt_name == 'rmsprop':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta=self.cli_args.beta, epsilon=epsilon)

        self.grad_b = None
        self.grad_w = None  

    #  Forward 
    def forward(self, X):
        if not hasattr(self, 'is_built') or not self.is_built:
            input_dim = X.shape[1] 
            output_dim = 10 
            
            layer_sizes = [input_dim] + self.hidden_size + [output_dim]
            self.layers = []
            
            for i in range(len(layer_sizes) - 1):
                is_output = (i == len(layer_sizes) - 2)
                self.layers.append(NeuralLayer(
                    input_size  = layer_sizes[i],
                    output_size = layer_sizes[i+1],
                    activation  = 'identity' if is_output else self.cli_args.activation,
                    weight_init = self.cli_args.weight_init,
                    layer_name  = 'output' if is_output else 'hidden'
                ))
            self.is_built = True

        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out # Returns raw logits

    def predict_proba(self, X):
        softmax_fn, _ = ACTIVATIONS['softmax']
        return softmax_fn(self.forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    #  Backward 
    def backward(self, y_true, y_pred_logits):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_w, grad_b as object arrays.
        grad_w[0] = gradient for output layer weights (shape: hidden_size, 10)
        grad_w[1] = gradient for first hidden layer weights (shape: 784, hidden_size)
        """
        m = y_pred_logits.shape[0]
        n = y_pred_logits.shape[1]

        y_true = np.array(y_true)
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_int = y_true.flatten().astype(int)
            y_onehot = np.zeros((m, n))
            y_onehot[np.arange(m), y_int] = 1.0
        else:
            y_onehot = y_true
        grad_w_list = []
        grad_b_list = []

        delta = self.loss_grad(y_onehot, y_pred_logits)
        for layer in reversed(self.layers):
            if layer.layer_name == 'output':
                m = layer.X.shape[0]
                layer.grad_w = (layer.X.T @ delta)
                layer.grad_b = np.sum(delta, axis=0, keepdims=True)
                delta = delta @ layer.W.T
            else:
                delta = layer.backward(delta)
            grad_w_list.append(layer.grad_w)
            grad_b_list.append(layer.grad_b)

        # Store as object arrays
        self.grad_w = np.empty(len(grad_w_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        
        for i, (gw, gb) in enumerate(zip(grad_w_list, grad_b_list)):
            self.grad_w[i] = gw  
            self.grad_b[i] = gb 
        
        return self.grad_w, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.layers)

    #  Evaluate 
    def evaluate(self, X, y_onehot, split_name='val'):
        logits = self.forward(X)
        if self.cli_args.loss == 'mse':
            loss   = self.loss(y_true=y_onehot, y_pred=logits)
            y_pred_lbl = np.argmax(logits, axis=1)
        else:
            probs  = ACTIVATIONS['softmax'][0](logits)
            loss   = self.loss(y_true=y_onehot, y_pred=probs)
            y_pred_lbl = np.argmax(probs, axis=1)

        y_true_lbl  = np.argmax(y_onehot, axis=1)
        metrics = compute_metrics(y_true_lbl, y_pred_lbl)
        metrics['loss'] = float(loss)
        return metrics

    #  Train 
    def train(self, X_train, y_train,
              epochs, batch_size, save_dir='.', wandb_run=None,
              track_grad_steps=50):

        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss' : [], 'val_acc' : [], 'val_f1' : [],
        }

        val_fraction = getattr(self.cli_args, 'val_fraction',
                               getattr(self.cli_args, 'val_split', 0.1))
        seed = getattr(self.cli_args, 'seed', 42)

        (X_tr, y_tr), (X_val, y_val) = train_val_split(
            X_train, y_train, val_fraction=val_fraction, seed=seed)

        n   = X_tr.shape[0]
        global_step = 0

        print(f"\n  {n} samples | batch={batch_size} | epochs={epochs} | "
              f"opt={self.cli_args.optimizer} | lr={self.cli_args.learning_rate}")

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuf = X_tr[idx]
            y_shuf = y_tr[idx]

            for start in range(0, n, batch_size):
                X_b = X_shuf[start : start + batch_size]
                y_b = y_shuf[start : start + batch_size]

                logits = self.forward(X_b)
                self.backward(y_true=y_b, y_pred_logits=logits)

                if global_step < track_grad_steps and self.layers[0].grad_w is not None:
                    per_neuron = np.linalg.norm(self.layers[0].grad_w, axis=0)
                    self.grad_history_layer0.append(per_neuron.copy())

                self.update_weights()
                global_step += 1

            # Per-epoch evaluation
            sample_idx = np.random.choice(n, size=min(5000, n), replace=False)
            train_m = self.evaluate(X_tr[sample_idx], y_tr[sample_idx], 'train')
            val_m  = self.evaluate(X_val, y_val, 'val')

            history['train_loss'].append(train_m['loss'])
            history['train_acc'].append(train_m['accuracy'])
            history['train_f1'].append(train_m['f1'])
            history['val_loss'].append(val_m['loss'])
            history['val_acc'].append(val_m['accuracy'])
            history['val_f1'].append(val_m['f1'])

            print(f"  [{epoch+1}/{epochs}]  "
                  f"Train Loss:{train_m['loss']:.4f} Acc:{train_m['accuracy']:.4f}  |  "
                  f"Val Loss:{val_m['loss']:.4f} Acc:{val_m['accuracy']:.4f} F1:{val_m['f1']:.4f}")

            if val_m['f1'] > self._best_val_f1:
                self._best_val_f1 = val_m['f1']
                self._best_epoch  = epoch + 1
                self.save_model(save_dir)

            if wandb_run is not None:
                log_dict = {
                    'epoch' : epoch + 1,
                    'train/loss' : train_m['loss'],
                    'train/accuracy' : train_m['accuracy'],
                    'train/precision': train_m['precision'],
                    'train/recall': train_m['recall'],
                    'train/f1' : train_m['f1'],
                    'val/loss' : val_m['loss'],
                    'val/accuracy': val_m['accuracy'],
                    'val/precision' : val_m['precision'],
                    'val/recall' : val_m['recall'],
                    'val/f1': val_m['f1'],
                }
                for li, layer in enumerate(self.layers[:-1]):
                    if layer.dead_neuron_counts:
                        log_dict[f'dead_neurons/layer_{li}'] = layer.dead_neuron_counts[-1]
                for li, layer in enumerate(self.layers):
                    if layer.grad_w is not None:
                        log_dict[f'grad_norm/layer_{li}'] = float(
                            np.linalg.norm(layer.grad_w))
                try:
                    wandb_run.log(log_dict)
                except Exception:
                    pass

        print(f"\n  Best val F1={self._best_val_f1:.4f} at epoch {self._best_epoch}\n")
        return history

    #  Get / Set Weights 
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f'W{i}'] = layer.W.copy()
            d[f'b{i}'] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """
        Modified to handle dynamic building during inference.
        """
        if not self.is_built:
            for i in range(len(weight_dict) // 2):
                w_key = f'W{i}'
                input_dim, output_dim = weight_dict[w_key].shape
                is_output = (f'W{i+1}' not in weight_dict)
                
                self.layers.append(NeuralLayer(
                    input_size=input_dim,
                    output_size=output_dim,
                    activation='identity' if is_output else self.cli_args.activation,
                    layer_name='output' if is_output else 'hidden'
                ))
            self.is_built = True

        # Standard loading logic
        for i, layer in enumerate(self.layers):
            if f'W{i}' in weight_dict:
                layer.W = weight_dict[f'W{i}'].copy()
            if f'b{i}' in weight_dict:
                layer.b = weight_dict[f'b{i}'].copy()

    #  Save / Load 
    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        weights_filename = getattr(self.cli_args, 'model_save_path', 'best_model.npy')
        weights_path = os.path.join(save_dir, weights_filename)
        np.save(weights_path, self.get_weights(), allow_pickle=True)

        # Config always saved alongside weights as best_config.json
        config_path = os.path.join(save_dir, 'best_config.json')
        cfg = {
            'hidden_size' : list(self.hidden_size),
            'num_layers': len(self.hidden_size),
            'activation': self.cli_args.activation,
            'weight_init' : self.cli_args.weight_init,
            'optimizer' : self.cli_args.optimizer,
            'learning_rate': float(self.cli_args.learning_rate),
            'weight_decay' : float(self.weight_decay),
            'loss': self.cli_args.loss,
            'epochs' : int(getattr(self.cli_args, 'epochs', 0)),
            'best_epoch': int(self._best_epoch),
            'best_val_f1' : float(self._best_val_f1),
            'dataset'  : getattr(self.cli_args, 'dataset', 'mnist'),
        }
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        print(f" Saved {weights_filename}  val_f1={self._best_val_f1:.4f}"
              f" epoch={self._best_epoch}")

    @classmethod
    def load(cls, weights_path, config_path):
        """
        Load a saved model.
        Usage: model = NeuralNetwork.load('best_model.npy', 'best_config.json')
        """
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        hidden_size = cfg.get('hidden_size', [128, 128, 64])
        cli_args = argparse.Namespace(
            hidden_size = hidden_size,
            num_layers  = cfg.get('num_layers', len(hidden_size)),
            activation  = cfg['activation'],
            weight_init = cfg['weight_init'],
            optimizer   = cfg['optimizer'],
            learning_rate   = cfg['learning_rate'],
            weight_deca = cfg.get('weight_decay', 0.0),
            loss = cfg['loss'],
            epochs = cfg.get('epochs', 0),
            dataset = cfg.get('dataset', 'mnist'),
            model_save_path = os.path.basename(weights_path),
            beta = 0.9,
            epsilon = 1e-8,
        )

        model = cls(cli_args)
        weight_dict = np.load(weights_path, allow_pickle=True).item()
        model.set_weights(weight_dict)
        model._best_val_f1 = cfg.get('best_val_f1', -1.0)
        model._best_epoch  = cfg.get('best_epoch',  0)

        print(f"Model loaded from '{weights_path}'")
        print(f"  Architecture : 784  "
              f"{'  '.join(str(n) for n in hidden_size)}  10")
        print(f"  Best val F1 : {cfg.get('best_val_f1', 'N/A')}  "
              f"(epoch {cfg.get('best_epoch', 'N/A')})")
        return model

    def layer_gradient_norms(self):
        return [float(np.linalg.norm(l.grad_w)) if l.grad_w is not None else 0.0
                for l in self.layers]