import numpy as np
import argparse
from ann.neural_network import NeuralNetwork
from ann.objective_functions import OBJECTIVE as LOSSES


CONFIG = {
    'beta': 0.9, 'epsilon': 1e-8,
    'adam_beta1': 0.9, 'adam_beta2': 0.999,
    'nadam_beta1': 0.9, 'nadam_beta2': 0.999,
}


def numerical_gradient(model, X, y_oh, layer_idx, param_name='W', eps=1e-5):
    layer  = model.layers[layer_idx]
    param  = getattr(layer, param_name)
    grad_num = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx       = it.multi_index
        orig      = param[idx]

        param[idx] = orig + eps
        probs_p    = model.predict_proba(X)          # probs not logits
        loss_p     = model.loss(y_true=y_oh, y_pred=probs_p)

        param[idx] = orig - eps
        probs_m    = model.predict_proba(X)
        loss_m     = model.loss(y_true=y_oh, y_pred=probs_m)

        grad_num[idx] = (loss_p - loss_m) / (2 * eps)
        param[idx]    = orig

        it.iternext()

    return grad_num


def check_gradients(model, X, y_oh, layer_idx=0, eps=1e-5, tol=1e-7):
    probs = model.predict_proba(X)
    model.backward(y_true=y_oh, y_pred=probs)

    grad_W_analytical = model.layers[layer_idx].grad_W.copy()
    grad_b_analytical = model.layers[layer_idx].grad_b.copy()

    grad_W_numerical  = numerical_gradient(model, X, y_oh, layer_idx, 'W', eps)
    grad_b_numerical  = numerical_gradient(model, X, y_oh, layer_idx, 'b', eps)

    diff_W   = np.max(np.abs(grad_W_analytical - grad_W_numerical))
    diff_b   = np.max(np.abs(grad_b_analytical - grad_b_numerical))
    max_diff = max(diff_W, diff_b)

    return max_diff, max_diff < tol


if __name__ == '__main__':
    np.random.seed(42)

    cli_args = argparse.Namespace(
        num_neurons   = [8, 8],
        activation    = 'sigmoid',
        weight_init   = 'xavier',
        optimizer     = 'sgd',
        learning_rate = 0.01,
        weight_decay  = 0.0,
        loss          = 'cross_entropy',
        model_save_path = 'best_model.npy',
        dataset       = 'mnist',
    )

    model = NeuralNetwork(CONFIG, cli_args)

    X   = np.random.randn(5, 784) * 0.1
    y   = np.array([0, 1, 2, 0, 1])
    y_oh = np.eye(10)[y]

    print(f"Gradient check — {len(model.layers)} layers\n")
    all_passed = True
    for i in range(len(model.layers)):
        max_diff, passed = check_gradients(model, X, y_oh, layer_idx=i)
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  Layer {i:2d} W  max_diff={max_diff:.2e}  {status}")
        all_passed = all_passed and passed

    print(f"\n{'All layers passed ✓' if all_passed else 'Some layers FAILED ✗'}")