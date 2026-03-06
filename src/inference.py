"""
Inference Script
Load a saved model and evaluate on the test set.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from utils.arguments import parse_arguments, BEST_MODEL_NPY, BEST_MODEL_CONFIG

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def load_model(model_path):
    """
    Load trained model weights from disk.
    Returns the weights dict — matches the spec: data = np.load(model_path, allow_pickle=True).item()
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_onehot, batch_size=512):
    """
    Evaluate model on test data.
    Returns dict: logits, accuracy, f1, precision, recall, confusion_matrix.
    """
    n          = X_test.shape[0]
    all_logits = []
    all_probs  = []
    for start in range(0, n, batch_size):
        X_b = X_test[start : start + batch_size]
        all_logits.append(model.forward(X_b))
        all_probs.append(model.predict_proba(X_b))

    logits     = np.vstack(all_logits)
    probs      = np.vstack(all_probs)
    y_pred_lbl = np.argmax(probs,    axis=1)
    y_true_lbl = np.argmax(y_onehot, axis=1)

    return {
        'logits' : logits,
        'accuracy' : accuracy_score(y_true_lbl, y_pred_lbl),
        'precision' : precision_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0),
        'recall' : recall_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0),
        'f1' : f1_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_lbl, y_pred_lbl),
    }


def main():
    """
    Main inference function.
    Returns dict: logits, accuracy, f1, precision, recall.
    """
    args = parse_arguments()

    weights_path = os.path.join(args.save_dir, args.model_save_path)
    config_path  = os.path.join(args.save_dir, args.config_path)

    # Load model
    model = NeuralNetwork.load(weights_path, config_path)

    # Load dataset
    (_, _), (X_test, y_test) = load_dataset(args.dataset)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, batch_size=args.batch_size)

    print('\n  Test Set Metrics')
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        print(f'  {k.capitalize():<12}: {results[k]:.4f}')

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    out = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in results.items()}
    out_path = os.path.join(args.save_dir, 'inference_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to '{out_path}'")

    # Optional W&B logging
    if not args.no_wandb and _WANDB_AVAILABLE:
        run = wandb.init(
            project = args.wandb_project,
            entity = args.wandb_entity,
            name = 'inference',
            config = {'dataset': args.dataset,
                       'model_path': args.model_save_path},
        )
        try:
            run.log({'test/accuracy' : results['accuracy'],
                     'test/f1' : results['f1'],
                     'test/precision': results['precision'],
                     'test/recall' : results['recall']})
        except Exception:
            pass
        run.finish()

    print('Evaluation complete!')
    return results


if __name__ == '__main__':
    main()