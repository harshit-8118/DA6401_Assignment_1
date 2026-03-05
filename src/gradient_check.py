import numpy as np
from ann.neural_network import NeuralNetwork
from utils.arguments import parse_arguments
from utils.data_loader import load_dataset

def numerical_gradient(model, X, y_oh, layer_idx, param_name='W', eps=1e-5):

    layer = model.layers[layer_idx]
    param = getattr(layer, param_name)

    grad_num = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'])

    while not it.finished:

        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        probs_p = model.predict_proba(X)
        loss_p = model.loss(y_true=y_oh, y_pred=probs_p)

        param[idx] = orig - eps
        probs_m = model.predict_proba(X)
        loss_m = model.loss(y_true=y_oh, y_pred=probs_m)

        grad_num[idx] = (loss_p - loss_m) / (2 * eps)

        param[idx] = orig
        it.iternext()

    return grad_num


def check_gradients(model, X, y_oh, layer_idx=0, eps=1e-5, tol=1e-7):

    probs = model.predict_proba(X)

    model.backward(y_true=y_oh, y_pred=probs)

    grad_W_analytical = model.layers[layer_idx].grad_w.copy()
    grad_b_analytical = model.layers[layer_idx].grad_b.copy()

    grad_W_numerical = numerical_gradient(model, X, y_oh, layer_idx, 'W', eps)
    grad_b_numerical = numerical_gradient(model, X, y_oh, layer_idx, 'b', eps)

    diff_W = np.max(np.abs(grad_W_analytical - grad_W_numerical))
    diff_b = np.max(np.abs(grad_b_analytical - grad_b_numerical))

    max_diff = max(diff_W, diff_b)

    return max_diff, max_diff < tol


if __name__ == "__main__":

    cli_args = parse_arguments()
    print(cli_args)

    model = NeuralNetwork(cli_args)

    (X, y), (_, _) = load_dataset(cli_args.dataset)

    X = X[:5, :]
    y_oh = y[:5, :]

    print(f"\nGradient check for {len(model.hidden_size)} layers\n")

    all_passed = True

    for i in range(len(model.hidden_size)):

        max_diff, passed = check_gradients(model, X, y_oh, layer_idx=i)

        status = "PASS" if passed else "FAIL"

        print(f"Layer {i}  max_diff={max_diff:.2e}  {status}")

        all_passed = all_passed and passed

    print("\nAll layers passed!" if all_passed else "\nGradient check FAILED")