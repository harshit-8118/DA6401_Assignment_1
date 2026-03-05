"""
Main Training Script
Entry point for training the MLP with command-line arguments.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import shutil
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from utils.arguments import parse_arguments, CONFIG, BEST_MODEL_NPY, BEST_MODEL_CONFIG
from utils.wandb_report import (
    log_5_samples_from_each_class, run_sweep, optimizer_showdown,
    vanishing_grad_analysis, dead_neuron_investigation,
    loss_function_comparison, global_performance_overlay_from_wandb,
    error_analysis, weight_init_symmetry, fashion_mnist_transfer,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── W&B helpers ───────────────────────────────────────────────────────────────

def make_wandb_run(args, name, group, extra_cfg=None):
    if args.no_wandb or not _WANDB_AVAILABLE:
        return None
    cfg = {
        'dataset'      : args.dataset,
        'epochs'       : args.epochs,
        'batch_size'   : args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer'    : args.optimizer,
        'activation'   : args.activation,
        'hidden_size'  : list(args.hidden_size),
        'num_layers'   : args.num_layers,
        'weight_init'  : args.weight_init,
        'loss'         : args.loss,
        'experiment'   : args.experiment,
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    return wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        name    = name,
        group   = group,
        config  = cfg,
    )


def _finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


# ── Best-model helpers ────────────────────────────────────────────────────────

def best_model_paths(args):
    return (os.path.join(args.save_dir, BEST_MODEL_NPY),
            os.path.join(args.save_dir, BEST_MODEL_CONFIG))


def run_training(args, x_train, y_train, x_test, y_test, wandb_run):
    """Train with current args, always saving to best_model.npy."""
    import argparse as _ap
    train_args = _ap.Namespace(**vars(args))
    train_args.model_save_path = BEST_MODEL_NPY

    model = NeuralNetwork(train_args)
    model.train(x_train, y_train,
                epochs=train_args.epochs,
                batch_size=train_args.batch_size,
                save_dir=train_args.save_dir,
                wandb_run=wandb_run)

    # Ensure best_config.json is present (save_model writes it automatically)
    test_m = model.evaluate(x_test, y_test, split_name='test')
    weights_path, config_path = best_model_paths(args)

    print(f"\n{'='*60}")
    print(f"  BEST MODEL  →  {weights_path}")
    print(f"  Val F1  : {model._best_val_f1:.4f}  (epoch {model._best_epoch})")
    print(f"  Test Acc: {test_m['accuracy']:.4f}   Test F1: {test_m['f1']:.4f}")
    print(f"{'='*60}\n")

    if wandb_run is not None and _WANDB_AVAILABLE:
        try:
            wandb_run.log({
                'test/loss'     : test_m['loss'],
                'test/accuracy' : test_m['accuracy'],
                'test/precision': test_m['precision'],
                'test/recall'   : test_m['recall'],
                'test/f1'       : test_m['f1'],
                'best/val_f1'   : model._best_val_f1,
                'best/epoch'    : model._best_epoch,
            })
        except Exception:
            pass
    return model, test_m


def load_best_model(args):
    weights_path, config_path = best_model_paths(args)
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f'\nBest model not found.\n'
            f'Run:  python train.py --experiment train\n'
            f'Expected: {weights_path}  {config_path}')
    return NeuralNetwork.load(weights_path, config_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Experiment : {args.experiment.upper()}')
    print(f'  Dataset    : {args.dataset}')
    print(f'  Arch       : {args.hidden_size}  ({args.num_layers} hidden layers)')
    print(f'  Project    : {args.wandb_project}')
    print(f'{"="*60}\n')

    # Load data for all other experiments
    dataset_name = ('fashion_mnist'
                    if args.experiment == 'fashion' else args.dataset)
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    exp = args.experiment
    
    if True:
        run = make_wandb_run(args, '2.0_train', '2.0_train')
        run_training(args, x_train, y_train, x_test, y_test, run)
        _finish(run)

    # Sweep is self-contained — load data and return immediately
    if args.experiment == 'sweep':
        if args.no_wandb or not _WANDB_AVAILABLE:
            print('W&B sweep requires wandb. Remove --no_wandb.')
            return
        (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)
        run_sweep(args, CONFIG, x_train, y_train, x_test, y_test,
                  NeuralNetwork=NeuralNetwork)
        return

    # 2.0 — train (saves best_model.npy + best_config.json)

    # 2.1 — sample visualisation
    if exp in ('visual', 'all'):
        run = make_wandb_run(args, '2.1_samples', '2.1_samples')
        log_5_samples_from_each_class(
            x_train, y_train, save_dir=args.save_dir, wandb_run=run)
        _finish(run)

    # 2.3 — optimizer showdown
    if exp in ('optimizer', 'all'):
        run = make_wandb_run(args, '2.3_optimizer_showdown',
                             '2.3_optimizer_showdown')
        run = optimizer_showdown(args, CONFIG, x_train, y_train,
                                 NeuralNetwork=NeuralNetwork, wandb_run=run)
        _finish(run)

    # 2.4 — vanishing gradient
    if exp in ('vanishing', 'all'):
        run = make_wandb_run(args, '2.4_vanishing_gradient',
                             '2.4_vanishing_gradient')
        run = vanishing_grad_analysis(args, CONFIG, x_train, y_train,
                                      NeuralNetwork=NeuralNetwork, wandb_run=run)
        _finish(run)

    # 2.5 — dead neurons
    if exp in ('dead', 'all'):
        run = make_wandb_run(args, '2.5_dead_neurons', '2.5_dead_neurons')
        run, _ = dead_neuron_investigation(args, CONFIG, x_train, y_train,
                                           NeuralNetwork=NeuralNetwork, wandb_run=run)
        _finish(run)

    # 2.6 — loss comparison
    if exp in ('loss', 'all'):
        run = make_wandb_run(args, '2.6_loss_comparison', '2.6_loss_comparison')
        run, _ = loss_function_comparison(args, CONFIG, x_train, y_train,
                                          NeuralNetwork=NeuralNetwork, wandb_run=run)
        _finish(run)

    # 2.7 — global overlay (needs prior sweep runs in W&B)
    if exp in ('overlay', 'all'):
        if args.no_wandb or not _WANDB_AVAILABLE:
            print('Overlay needs W&B data — enable W&B and run sweep first.')
        else:
            run = make_wandb_run(args, '2.7_global_overlay', '2.7_global_overlay')
            global_performance_overlay_from_wandb(args, wandb_run=run)
            _finish(run)

    # 2.8 — error analysis (loads best_model.npy)
    if exp in ('error', 'all'):
        try:
            model = load_best_model(args)
        except FileNotFoundError as e:
            print(e)
            if exp != 'all':
                return
        else:
            run = make_wandb_run(args, '2.8_error_analysis', '2.8_error_analysis')
            error_analysis(model, x_test, y_test,
                           dataset_name=args.dataset,
                           save_dir=args.save_dir, wandb_run=run)
            _finish(run)

    # 2.9 — weight init symmetry
    if exp in ('symmetry', 'all'):
        run = make_wandb_run(args, '2.9_weight_init_symmetry',
                             '2.9_weight_init_symmetry')
        run, _ = weight_init_symmetry(args, CONFIG, x_train, y_train,
                                      NeuralNetwork=NeuralNetwork, wandb_run=run,
                                      n_neurons_to_track=5, track_grad_steps=50)
        _finish(run)

    # 2.10 — fashion-MNIST transfer
    if exp in ('fashion', 'all'):
        (x_f_train, y_f_train), (x_f_test, y_f_test) = load_dataset('fashion_mnist')
        run = make_wandb_run(args, '2.10_fashion_transfer',
                             '2.10_fashion_transfer')
        run, _ = fashion_mnist_transfer(args, CONFIG,
                                        x_f_train, y_f_train,
                                        x_f_test, y_f_test,
                                        NeuralNetwork=NeuralNetwork, wandb_run=run)
        _finish(run)


if __name__ == '__main__':
    main()