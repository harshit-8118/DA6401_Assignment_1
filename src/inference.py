import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

import matplotlib
matplotlib.use('Agg')
import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from ann  import NeuralNetwork
from utils import load_dataset, parse_arguments, CONFIG


def load_model(model_path, config_path=None):
    if config_path is None:
        config_path = model_path.replace('.npy', '_config.json')
    return NeuralNetwork.load(model_path, config_path, CONFIG)


def evaluate_model(model, X_test, y_onehot, batch_size=512):
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
        'accuracy'        : accuracy_score(y_true_lbl, y_pred_lbl),
        'precision'       : precision_score(y_true_lbl, y_pred_lbl,
                                            average='macro', zero_division=0),
        'recall'          : recall_score(y_true_lbl, y_pred_lbl,
                                         average='macro', zero_division=0),
        'f1'              : f1_score(y_true_lbl, y_pred_lbl,
                                     average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_lbl, y_pred_lbl),
        'logits'          : logits,
    }


def main():
    args         = parse_arguments()
    weights_path = os.path.join(args.save_dir, args.model_save_path)
    config_path  = os.path.join(args.save_dir, args.config_path)

    #  W&B 
    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = args.wandb_run_name,
            config  = {'dataset': args.dataset, 'model_path': args.model_save_path},
        )

    #  Load model 
    model = load_model(weights_path, config_path)

    #  Load data 
    (X_train, y_train), (X_test, y_test) = load_dataset(args.dataset)

    #  Train metrics 
    train_r = evaluate_model(model, X_train, y_train, args.batch_size)
    print('\n Train Set Metrics ')
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        print(f'  {k.capitalize():<12}: {train_r[k]:.4f}')

    #  Test metrics 
    test_r = evaluate_model(model, X_test, y_test, args.batch_size)
    print('\n Test Set Metrics ')
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        print(f'  {k.capitalize():<12}: {test_r[k]:.4f}')

    #  Save JSON 
    out = {k: (v.tolist() if hasattr(v, 'tolist') else v)
           for k, v in test_r.items()}
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'inference_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to '{args.save_dir}/inference_results.json'")

    if wandb_run is not None:
        wandb_run.finish()

    print("Evaluation complete!")
    return test_r


if __name__ == '__main__':
    main()