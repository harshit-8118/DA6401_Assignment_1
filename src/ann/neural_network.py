"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from .neural_layer import NeuralLayer
from utils import train_val_split
from .objective_functions import OBJECTIVE
from .activations import ACTIVATIONS
from .optimizers import OPTIMIZERS
import numpy as np
import json
import os
import argparse


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true_labels, y_pred_labels, num_classes=10):
    accuracy = float(np.mean(y_true_labels == y_pred_labels))

    precision_per_class = np.zeros(num_classes)
    recall_per_class    = np.zeros(num_classes)

    for c in range(num_classes):
        tp = np.sum((y_pred_labels == c) & (y_true_labels == c))
        fp = np.sum((y_pred_labels == c) & (y_true_labels != c))
        fn = np.sum((y_pred_labels != c) & (y_true_labels == c))
        precision_per_class[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per_class[c]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precision = float(np.mean(precision_per_class))
    recall    = float(np.mean(recall_per_class))
    denom     = precision + recall
    f1        = float(2 * precision * recall / denom) if denom > 0 else 0.0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# ── NeuralNetwork ─────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, config, cli_args):
        self.config       = config
        self.cli_args     = cli_args
        self.layers       = []
        self.weight_decay = cli_args.weight_decay

        self.loss, self.loss_grad = OBJECTIVE[cli_args.loss]
        self._best_val_f1 = -1.0
        self._best_epoch  = 0

        # ── Per-step gradient history for first hidden layer neurons (2.9) ──
        self.grad_history_layer0 = []

        layer_sizes = [784] + list(cli_args.num_neurons)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NeuralLayer(
                input_size  = layer_sizes[i],
                output_size = layer_sizes[i + 1],
                activation  = cli_args.activation,
                weight_init = cli_args.weight_init,
                layer_name  = 'hidden',
            ))

        self.layers.append(NeuralLayer(
            input_size  = layer_sizes[-1],
            output_size = 10,
            activation  = 'identity',
            weight_init = cli_args.weight_init,
            layer_name  = 'output',
        ))

        opt_name    = cli_args.optimizer
        opt_cls     = OPTIMIZERS[opt_name]
        self.is_nag = (opt_name == 'nag')

        if opt_name == 'sgd':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate)
        elif opt_name in ('momentum', 'nag'):
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta=config['beta'])
        elif opt_name == 'rmsprop':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta=config['beta'],
                                     epsilon=config['epsilon'])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied).
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out   # raw logits

    def predict_proba(self, X):
        softmax_fn, _ = ACTIVATIONS['softmax']
        return softmax_fn(self.forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - grad_Ws[0] is gradient for the last (output) layer weights,
          grad_bs[0] is gradient for the last layer biases, and so on.
        """
        loss_val = self.loss(y_true=y_true, y_pred=y_pred)
        delta    = self.loss_grad(y_true=y_true, y_pred=y_pred)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=self.weight_decay)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # index 0 = last (output) layer, as per template
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return loss_val

    def update_weights(self):
        self.optimizer.update(self.layers)

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, X, y_onehot, split_name='val'):
        logits = self.forward(X)

        if self.cli_args.loss == 'mse':
            loss       = self.loss(y_true=y_onehot, y_pred=logits)
            y_pred_lbl = np.argmax(logits, axis=1)
        else:
            probs      = ACTIVATIONS['softmax'][0](logits)
            loss       = self.loss(y_true=y_onehot, y_pred=probs)
            y_pred_lbl = np.argmax(probs, axis=1)

        y_true_lbl      = np.argmax(y_onehot, axis=1)
        metrics         = compute_metrics(y_true_lbl, y_pred_lbl)
        metrics['loss'] = float(loss)
        return metrics

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, X_train, y_train,
              epochs, batch_size, save_dir='.', wandb_run=None,
              track_grad_steps=50):

        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss'  : [], 'val_acc' : [], 'val_f1'  : [],
        }

        (X_train, y_train), (X_val, y_val) = train_val_split(
            X_train, y_train, val_fraction=self.config['val_split'], seed=self.cli_args.seed)
        n           = X_train.shape[0]
        global_step = 0

        print(f"\n{'─'*80}")
        print(f"  {n} samples | batch={batch_size} | epochs={epochs} | "
              f"opt={self.cli_args.optimizer} | lr={self.cli_args.learning_rate} | "
              f"wd={self.weight_decay}")
        print(f"{'─'*80}")

        for epoch in range(epochs):
            idx    = np.random.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            for start in range(0, n, batch_size):
                X_b = X_shuf[start : start + batch_size]
                y_b = y_shuf[start : start + batch_size]

                if self.is_nag:
                    self.optimizer.lookahead(self.layers)

                logits = self.forward(X_b)
                if self.cli_args.loss == 'mse':
                    loss = self.backward(y_true=y_b, y_pred=logits)
                else:
                    probs = self.predict_proba(X_b)
                    loss  = self.backward(y_true=y_b, y_pred=probs)

                if self.is_nag:
                    self.optimizer.restore(self.layers)

                if global_step < track_grad_steps and self.layers[0].grad_W is not None:
                    per_neuron = np.linalg.norm(self.layers[0].grad_W, axis=0)
                    self.grad_history_layer0.append(per_neuron.copy())

                self.update_weights()
                global_step += 1

            # ── Per-epoch evaluation ───────────────────────────────────────────
            sample_idx = np.random.choice(n, size=min(5000, n), replace=False)
            train_m    = self.evaluate(X_train[sample_idx], y_train[sample_idx], 'train')
            val_m      = self.evaluate(X_val, y_val, 'val')

            history['train_loss'].append(train_m['loss'])
            history['train_acc'].append(train_m['accuracy'])
            history['train_f1'].append(train_m['f1'])
            history['val_loss'].append(val_m['loss'])
            history['val_acc'].append(val_m['accuracy'])
            history['val_f1'].append(val_m['f1'])

            print('─' * 50)
            print(f"epoch [{epoch+1}/{epochs}]")
            print(f"Train  Loss:{train_m['loss']:.6f}  Acc:{train_m['accuracy']:.4f}  F1:{train_m['f1']:.4f}")
            print(f"Valid  Loss:{val_m['loss']:.6f}  Acc:{val_m['accuracy']:.4f}  F1:{val_m['f1']:.4f}")

            if wandb_run is not None:
                log_dict = {
                    'epoch'           : epoch + 1,
                    'train/loss'      : train_m['loss'],
                    'train/accuracy'  : train_m['accuracy'],
                    'train/precision' : train_m['precision'],
                    'train/recall'    : train_m['recall'],
                    'train/f1'        : train_m['f1'],
                    'val/loss'        : val_m['loss'],
                    'val/accuracy'    : val_m['accuracy'],
                    'val/precision'   : val_m['precision'],
                    'val/recall'      : val_m['recall'],
                    'val/f1'          : val_m['f1'],
                }

                for li, layer in enumerate(self.layers[:-1]):
                    if layer.dead_neuron_counts:
                        log_dict[f'dead_neurons/layer_{li}'] = layer.dead_neuron_counts[-1]

                for li, layer in enumerate(self.layers):
                    if layer.grad_W is not None:
                        log_dict[f'grad_norm/layer_{li}'] = float(np.linalg.norm(layer.grad_W))

                wandb_run.log(log_dict)

            if val_m['f1'] > self._best_val_f1:
                self._best_val_f1 = val_m['f1']
                self._best_epoch  = epoch + 1
                self.save_model(save_dir)

        print(f"\n  Best val F1={self._best_val_f1:.4f} at epoch {self._best_epoch}")
        print(f"{'─'*80}\n")

        return history

    # ── Get / Set Weights ─────────────────────────────────────────────────────

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        weights_filename = getattr(self.cli_args, 'model_save_path', 'best_model.npy')
        np.save(os.path.join(save_dir, weights_filename), self.get_weights(), allow_pickle=True)

        best_config = {
            'num_neurons'  : list(self.cli_args.num_neurons),
            'activation'   : self.cli_args.activation,
            'weight_init'  : self.cli_args.weight_init,
            'optimizer'    : self.cli_args.optimizer,
            'learning_rate': self.cli_args.learning_rate,
            'weight_decay' : float(self.weight_decay),
            'loss'         : self.cli_args.loss,
            'best_val_f1'  : float(self._best_val_f1),
            'best_epoch'   : int(self._best_epoch),
            'dataset'      : getattr(self.cli_args, 'dataset', 'mnist'),
        }

        config_filename = weights_filename.replace('.npy', '_config.json')
        with open(os.path.join(save_dir, config_filename), 'w') as f:
            json.dump(best_config, f, indent=2)

        print(f"   → Saved {weights_filename} + {config_filename} "
              f"(val_f1={self._best_val_f1:.4f}, epoch={self._best_epoch})")

    @classmethod
    def load(cls, weights_path, config_path, hyperparams_config):
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        cli_args = argparse.Namespace(
            num_neurons     = cfg['num_neurons'],
            activation      = cfg['activation'],
            weight_init     = cfg['weight_init'],
            optimizer       = cfg['optimizer'],
            learning_rate   = cfg['learning_rate'],
            weight_decay    = cfg['weight_decay'],
            loss            = cfg['loss'],
            model_save_path = os.path.basename(weights_path),
            dataset         = cfg.get('dataset', 'mnist'),
        )

        model  = cls(hyperparams_config, cli_args)
        weight_dict = np.load(weights_path, allow_pickle=True).item() 
        model.set_weights(weight_dict) 

        print(f"Model loaded from '{weights_path}'")
        print(f"  Architecture : 784 → {' → '.join(str(n) for n in cfg['num_neurons'])} → 10")
        print(f"  Best val F1  : {cfg.get('best_val_f1', 'N/A')}")
        return model

    def layer_gradient_norms(self):
        return [float(np.linalg.norm(l.grad_W)) if l.grad_W is not None else 0.0
                for l in self.layers]