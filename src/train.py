import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import numpy as np
import shutil

from ann  import NeuralNetwork
from utils import (
    load_dataset, 
    log_5_samples_from_each_class,
    optimizer_showdown,
    vanishing_grad_analysis,
    run_sweep,
    dead_neuron_investigation,
    loss_function_comparison,
    global_performance_overlay_from_wandb,
    error_analysis,
    weight_init_symmetry,
    fashion_mnist_transfer,
    parse_arguments, CONFIG, BEST_MODEL_NPY, BEST_MODEL_CONFIG
)
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


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
        'num_neurons'  : list(args.num_neurons),
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
        reinit  = True,
    )


def best_model_path(args):
    """Always returns the canonical best-model paths under save_dir."""
    return (
        os.path.join(args.save_dir, BEST_MODEL_NPY),
        os.path.join(args.save_dir, BEST_MODEL_CONFIG),
    )


def run_best_model_training(args, CONFIG, x_train, y_train,
                              x_test, y_test, wandb_run):
    train_args = argparse.Namespace(**vars(args))
    train_args.model_save_path = BEST_MODEL_NPY   # force canonical name

    model = NeuralNetwork(CONFIG, train_args)
    model.train(x_train, y_train,
                epochs=train_args.epochs,
                batch_size=train_args.batch_size,
                save_dir=train_args.save_dir,
                wandb_run=wandb_run)
    src = os.path.join(train_args.save_dir, BEST_MODEL_NPY.replace('.npy', '_config.json'))
    dst = os.path.join(train_args.save_dir, BEST_MODEL_CONFIG)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    test_m = model.evaluate(x_test, y_test, split_name='test')

    weights_path, config_path = best_model_path(args)
    print(f"\n{'='*60}")
    print(f"  BEST MODEL SAVED")
    print(f"  Weights : {weights_path}")
    print(f"  Config  : {config_path}")
    print(f"  Val  F1 : {model._best_val_f1:.4f}  (epoch {model._best_epoch})")
    print(f"  Test Acc: {test_m['accuracy']:.4f}   Test F1: {test_m['f1']:.4f}")
    print(f"{'='*60}\n")

    if wandb_run is not None and _WANDB_AVAILABLE:
        wandb_run.log({
            'test/loss'     : test_m['loss'],
            'test/accuracy' : test_m['accuracy'],
            'test/precision': test_m['precision'],
            'test/recall'   : test_m['recall'],
            'test/f1'       : test_m['f1'],
            'best/val_f1'   : model._best_val_f1,
            'best/epoch'    : model._best_epoch,
        })

    return model, test_m


def load_best_model(args, CONFIG):
    weights_path, config_path = best_model_path(args)

    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\nBest model not found at: {weights_path}\n or \nBest config not found at: {config_path}\n"
        )
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    print(f"\n{'='*60}")
    print(f"  LOADING BEST MODEL")
    print(f"  Weights     : {weights_path}")
    print(f"  Architecture: 784 -> {' -> '.join(str(n) for n in cfg['num_neurons'])} -> 10")
    print(f"  Optimizer   : {cfg['optimizer']}  LR: {cfg['learning_rate']}")
    print(f"  Activation  : {cfg['activation']}")
    print(f"  Saved Val F1: {cfg.get('best_val_f1', 'N/A')}  (epoch {cfg.get('best_epoch', 'N/A')})")
    print(f"{'='*60}\n")

    model = NeuralNetwork.load(weights_path, config_path, CONFIG)
    return model


def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Experiment : {args.experiment.upper()}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Entity     : {args.wandb_entity}")
    print(f"  Project    : {args.wandb_project}")
    print(f"{'='*60}\n")

    dataset_name = 'fashion_mnist' if args.experiment == 'fashion' else args.dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)

    exp = args.experiment

    # 2.0 training
    if exp in ('train', 'all'):
        wandb_run = make_wandb_run(args, '2.0_train_best_model', '2.0_train_best_model')
        run_best_model_training(args, CONFIG, x_train, y_train,
                                  x_test, y_test, wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.1  Sample Visualisation 
    if exp in ('visual', 'all'):
        wandb_run = make_wandb_run(args, '2.1_samples', '2.1_samples')
        log_5_samples_from_each_class(
            x_train, y_train, save_dir=args.save_dir, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.2  Sweep 
    if exp == 'sweep':
        if args.no_wandb or not _WANDB_AVAILABLE:
            print("W&B sweep requires wandb. Remove --no_wandb.")
            return
        run_sweep(args, CONFIG, x_train, y_train, NeuralNetwork=NeuralNetwork)

    #  2.3  Optimizer Showdown 
    if exp in ('optimizer', 'all'):
        wandb_run = make_wandb_run(args, '2.3_optimizer_showdown',
                                   '2.3_optimizer_showdown')
        optimizer_showdown(args, CONFIG, x_train, y_train, 
                           NeuralNetwork=NeuralNetwork, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.4  Vanishing Gradient 
    if exp in ('vanishing', 'all'):
        wandb_run = make_wandb_run(args, '2.4_vanishing_gradient',
                                   '2.4_vanishing_gradient')
        vanishing_grad_analysis(args, CONFIG, x_train, y_train, 
                                NeuralNetwork=NeuralNetwork, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.5  Dead Neurons 
    if exp in ('dead', 'all'):
        wandb_run = make_wandb_run(args, '2.5_dead_neurons', '2.5_dead_neurons')
        dead_neuron_investigation(args, CONFIG, x_train, y_train, 
                                  NeuralNetwork=NeuralNetwork, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.6  Loss Comparison 
    if exp in ('loss', 'all'):
        wandb_run = make_wandb_run(args, '2.6_loss_comparison', '2.6_loss_comparison')
        loss_function_comparison(args, CONFIG, x_train, y_train,
                                 NeuralNetwork=NeuralNetwork, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.7  Overlay 
    if exp in ('overlay', 'all'):
        if args.no_wandb or not _WANDB_AVAILABLE:
            print("Overlay pulls data from W&B. Run after sweep with W&B enabled.")
        else:
            wandb_run = make_wandb_run(args, '2.7_global_overlay', '2.7_global_overlay')
            global_performance_overlay_from_wandb(args, wandb_run=wandb_run)
            if wandb_run is not None:
                wandb_run.finish()

    #  2.8  Error Analysis — ALWAYS loads best_model.npy 
    if exp in ('error', 'all'):
        try:
            model = load_best_model(args, CONFIG)
        except FileNotFoundError as e:
            print(e)
            if exp == 'all':
                print("  Skipping error analysis — run --experiment train first.")
            return

        wandb_run = make_wandb_run(args, '2.8_error_analysis', '2.8_error_analysis')
        error_analysis(model, x_test, y_test,
                       dataset_name=args.dataset,
                       save_dir=args.save_dir,
                       wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.9  Weight Init Symmetry 
    if exp in ('symmetry', 'all'):
        wandb_run = make_wandb_run(args, '2.9_weight_init_symmetry',
                                   '2.9_weight_init_symmetry')
        weight_init_symmetry(args, CONFIG, x_train, y_train,
                             NeuralNetwork=NeuralNetwork,
                             wandb_run=wandb_run,
                             n_neurons_to_track=5,
                             track_grad_steps=50)
        if wandb_run is not None:
            wandb_run.finish()

    #  2.10  Fashion Transfer 
    if exp in ('fashion', 'all'):
        (x_f_train, y_f_train), (x_f_test, y_f_test) = load_dataset('fashion_mnist')
        wandb_run = make_wandb_run(args, '2.10_fashion_transfer', '2.10_fashion_transfer')
        fashion_mnist_transfer(args, CONFIG,
                               x_f_train, y_f_train, 
                               x_f_test, y_f_test,
                               NeuralNetwork=NeuralNetwork,
                               wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()