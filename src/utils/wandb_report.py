"""
utils/wandb_report.py
Experiment helpers for W&B report sections 2.1 – 2.10.
"""

import argparse
import os
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.data_loader import load_dataset, train_val_split
from utils.plots_fig import (
    plot_dead_bar, plot_dead_dist, plot_grad_comparison,
    plot_loss_comparison, plot_val_accuracy,
    creative_failure_viz, plot_symmetry,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def make_args(base_args, overrides: dict) -> argparse.Namespace:
    """Copy base_args and apply overrides.  Keeps hidden_size in sync."""
    d = vars(base_args).copy()
    d.update(overrides)
    ns = argparse.Namespace(**d)
    # keep hidden_size consistent with hidden_size / overridden list
    if 'hidden_size' in overrides:
        ns.hidden_size = list(overrides['hidden_size'])
        ns.num_layers  = len(ns.hidden_size)
    elif 'hidden_size' in overrides:
        ns.hidden_size = list(overrides['hidden_size'])
        ns.num_layers  = len(ns.hidden_size)
    return ns


def plots_dir(save_dir: str) -> str:
    p = os.path.join(save_dir, 'plots')
    os.makedirs(p, exist_ok=True)
    return p


def save_and_log(fig, filename, save_dir, wandb_key, wandb_run=None, caption=''):
    out_path = os.path.join(plots_dir(save_dir), filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved -> {out_path}')
    if wandb_run is not None and _WANDB_AVAILABLE:
        try:
            wandb_run.log({wandb_key: wandb.Image(out_path, caption=caption)})
            print(f"  W&B logged key='{wandb_key}'")
        except Exception as exc:
            print(f'  W&B log skipped ({exc})')
    return out_path


def _begin_sub_run(args, name: str, group: str, extra_cfg: dict = None):
    """Start an isolated sub-run (per optimizer / activation / etc.)."""
    if not _WANDB_AVAILABLE or getattr(args, 'no_wandb', True):
        return None
    cfg = {
        'dataset'      : getattr(args, 'dataset',       'mnist'),
        'epochs'       : getattr(args, 'epochs',        10),
        'batch_size'   : getattr(args, 'batch_size',    32),
        'learning_rate': getattr(args, 'learning_rate', 0.01),
        'optimizer'    : getattr(args, 'optimizer',     'momentum'),
        'activation'   : getattr(args, 'activation',   'relu'),
        'hidden_size'  : list(getattr(args, 'hidden_size', [])),
        'num_layers'   : getattr(args, 'num_layers',    3),
        'weight_init'  : getattr(args, 'weight_init',  'xavier'),
        'loss'         : getattr(args, 'loss',          'cross_entropy'),
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    return wandb.init(
        project = getattr(args, 'wandb_project', 'DA6401_assignment1v6'),
        entity  = getattr(args, 'wandb_entity',  None),
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


def restore_outer(wandb_run):
    """Re-open the outer run after sub-run loops may have invalidated it."""
    if wandb_run is None or not _WANDB_AVAILABLE:
        return wandb_run
    try:
        if getattr(wandb_run, '_is_finished', False):
            wandb_run = wandb.init(
                id      = wandb_run.id,
                project = wandb_run.project,
                entity  = wandb_run.entity,
                resume  = 'allow',
            )
    except Exception:
        pass
    return wandb_run


# ── 2.1 ──────────────────────────────────────────────────────────────────────

def log_5_samples_from_each_class(X, y, image_shape=(28, 28),
                                   save_dir='.', wandb_run=None):
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    classes   = np.unique(y)
    fig, axes = plt.subplots(len(classes), 5, figsize=(10, 2 * len(classes)))
    if len(classes) == 1:
        axes = np.expand_dims(axes, 0)
    wandb_images = []
    for i, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        sel = np.random.choice(idx, size=min(5, len(idx)), replace=False)
        for j in range(5):
            ax = axes[i, j]; ax.axis('off')
            if j < len(sel):
                img = X[sel[j]]
                if img.ndim == 1:
                    img = img.reshape(image_shape)
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Class {cls}', fontsize=8)
                if wandb_run is not None and _WANDB_AVAILABLE:
                    wandb_images.append(
                        wandb.Image(img, caption=f'Class {cls} | sample {j+1}'))
    plt.suptitle('2.1 - 5 Samples per Class', fontsize=11)
    plt.tight_layout()
    save_and_log(fig, '2.1_samples_grid.png', save_dir,
                 '2.1/samples_grid', wandb_run)
    if wandb_run is not None and _WANDB_AVAILABLE and wandb_images:
        try:
            wandb_run.log({'2.1/samples_gallery': wandb_images})
        except Exception:
            pass
    plt.close(fig)


# ── 2.2 sweep ────────────────────────────────────────────────────────────────

def run_sweep(args, CONFIG, x_train, y_train, x_test, y_test,
              NeuralNetwork=None):
    """
    W&B Bayesian sweep — 100 runs.
    Uses load_dataset splits directly:
      model.train receives x_train + y_train (does its own internal val split).
      x_val / y_val not passed to model — reserved for external consistency checks.
      x_test / y_test used to log test metrics after each run.
    """
    if not _WANDB_AVAILABLE:
        raise RuntimeError('wandb is required for run_sweep')

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val/f1', 'goal': 'maximize'},
        'parameters': {
            'epochs'       : {'values': [10, 20, 30, 50]},
            'batch_size'   : {'values': [32, 64, 128]},
            'learning_rate': {'values': [0.0001, 0.001, 0.005, 0.01]},
            'optimizer'    : {'values': ['momentum', 'nag', 'xavier', 'rmsprop']},
            'hidden_size'  : {'values': [[64, 64], [128, 128],
                                         [128, 128, 64], [128, 128, 128]]},
            'activation'   : {'values': ['relu', 'tanh', 'sigmoid']},
            'weight_init'  : {'values': ['random', 'xavier']},
            'weight_decay' : {'values': [0.0, 0.001, 0.0005]},
            'loss'         : {'values': ['cross_entropy', 'mse']},
        },
    }

    def sweep_train():
        run = wandb.init()
        cfg = wandb.config

        hidden_size = list(cfg.hidden_size)
        sweep_args  = argparse.Namespace(
            dataset        = args.dataset,
            epochs         = cfg.epochs,
            batch_size     = cfg.batch_size,
            learning_rate  = cfg.learning_rate,
            optimizer      = cfg.optimizer,
            loss           = cfg.loss,
            weight_decay   = cfg.weight_decay,
            hidden_size    = hidden_size,
            num_layers     = len(hidden_size),
            activation     = cfg.activation,
            weight_init    = cfg.weight_init,
            beta           = 0.9,
            epsilon        = 1e-8,
            val_fraction   = CONFIG.get('val_split', 0.1),
            seed           = 42,
            save_dir       = 'sweep_models',
            model_save_path= f'sweep_{run.id}.npy',
        )
        run.name = (f"{cfg.optimizer}_lr{cfg.learning_rate}"
                    f"_{cfg.activation}_b{cfg.batch_size}")
        np.random.seed(42)

        # Train using NeuralNetwork.train — it handles its own val split internally
        model = NeuralNetwork(sweep_args)
        model.train(x_train, y_train,
                    epochs=sweep_args.epochs,
                    batch_size=sweep_args.batch_size,
                    save_dir=sweep_args.save_dir,
                    wandb_run=run)

        # Log test metrics on the real held-out test set
        test_m = model.evaluate(x_test, y_test, split_name='test')
        try:
            run.log({
                'test/accuracy' : test_m['accuracy'],
                'test/f1'       : test_m['f1'],
                'test/loss'     : test_m['loss'],
            })
        except Exception:
            pass
        run.finish()

    sweep_id = wandb.sweep(
        sweep_config,
        project = getattr(args, 'wandb_project', 'DA6401_assignment1v6'),
        entity  = getattr(args, 'wandb_entity',  None),
    )
    print(f'Sweep ID: {sweep_id}')
    wandb.agent(sweep_id, function=sweep_train, count=100)


# ── 2.3 optimizer showdown ───────────────────────────────────────────────────

def optimizer_showdown(args, CONFIG, x_train, y_train,
                        NeuralNetwork=None, wandb_run=None):
    OPTS    = ['sgd', 'momentum', 'nag', 'rmsprop']
    summary = {}

    for opt in OPTS:
        print(f'\nOptimizer Showdown: {opt.upper()}')
        run_args = make_args(args, {
            'optimizer'       : opt,
            'hidden_size'     : [128, 128, 128],
            'activation'      : 'relu',
            'epochs'          : 10,
            'weight_decay'    : 0.0,
            'model_save_path' : f'showdown_{opt}.npy',
        })
        sub   = _begin_sub_run(run_args, f'showdown_{opt}',
                               '2.3_optimizer_showdown',
                               {'experiment': '2.3_optimizer_showdown'})
        model = NeuralNetwork(run_args)
        model.train(x_train, y_train,
                    epochs=run_args.epochs, batch_size=run_args.batch_size,
                    save_dir=run_args.save_dir, wandb_run=sub)
        summary[opt] = model._best_val_f1
        _finish(sub)

    wandb_run = restore_outer(wandb_run)
    fig, ax   = plt.subplots(figsize=(9, 4))
    vals      = [summary[o] for o in OPTS]
    # Use matplotlib tab10 colors or a custom muted palette
    muted_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [muted_colors[i % len(muted_colors)] for i in range(len(OPTS))]
    bars      = ax.bar(OPTS, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Best Val F1')
    ax.set_title('2.3 - Optimizer Showdown: Best Validation F1')
    plt.tight_layout()
    save_and_log(fig, '2.3_optimizer_showdown_summary.png', args.save_dir,
                 '2.3/optimizer_showdown_summary', wandb_run)
    plt.close(fig)
    return wandb_run


# ── 2.4 vanishing gradient ───────────────────────────────────────────────────

def vanishing_grad_analysis(args, CONFIG, x_train, y_train,
                             NeuralNetwork=None, wandb_run=None):
    ACTS    = ['sigmoid', 'relu']
    CONFIGS = {
        'shallow': [128],
        'medium' : [128, 128, 128],
        'deep'   : [128, 128, 128, 128, 128],
    }
    summary = {d: {} for d in CONFIGS}

    for depth_name, neurons in CONFIGS.items():
        for act in ACTS:
            label    = f'{depth_name}_{act}'
            run_args = make_args(args, {
                'activation'      : act,
                'optimizer'       : 'rmsprop',
                'learning_rate'   : 0.001,
                'epochs'          : 15,
                'hidden_size'     : neurons,
                'model_save_path' : f'vanishing_{label}.npy',
            })
            sub   = _begin_sub_run(run_args, f'vanishing_{label}',
                                   '2.4_vanishing_gradient_analysis',
                                   {'experiment': '2.4', 'depth': depth_name})
            model = NeuralNetwork(run_args)
            model.train(x_train, y_train,
                        epochs=run_args.epochs, batch_size=run_args.batch_size,
                        save_dir=run_args.save_dir, wandb_run=sub)
            summary[depth_name][act] = model.layer_gradient_norms()
            _finish(sub)

    wandb_run = restore_outer(wandb_run)
    palette   = {'sigmoid': '#e74c3c', 'relu': '#2ecc71'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (depth_name, neurons) in zip(axes, CONFIGS.items()):
        n_layers = max(len(summary[depth_name][a]) for a in ACTS)
        x_pos, width = np.arange(n_layers), 0.35
        for i, act in enumerate(ACTS):
            norms = summary[depth_name].get(act, [0] * n_layers)
            ax.bar(x_pos + i * width, norms, width,
                   color=palette[act], alpha=0.85, label=act,
                   edgecolor='k', linewidth=0.6)
        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
        ax.set_title(f'{depth_name.capitalize()} ({len(neurons)} hidden)')
        ax.set_ylabel('Gradient Norm'); ax.legend()
    fig.suptitle('2.4 - Vanishing Gradient (Sigmoid vs ReLU)', fontsize=12)
    plt.tight_layout()
    save_and_log(fig, '2.4_vanishing_gradient_summary.png', args.save_dir,
                 '2.4/vanishing_gradient_summary', wandb_run)
    plt.close(fig)
    return wandb_run


# ── 2.5 dead neuron ──────────────────────────────────────────────────────────

def dead_neuron_investigation(args, CONFIG, x_train, y_train,
                               NeuralNetwork=None, wandb_run=None):
    high_lr        = getattr(args, 'high_lr', 0.1)
    configs_to_run = [
        {'activation': 'relu',    'learning_rate': high_lr, 'tag': 'relu_high_lr'},
        {'activation': 'tanh',    'learning_rate': high_lr, 'tag': 'tanh_high_lr'},
        {'activation': 'sigmoid', 'learning_rate': high_lr, 'tag': 'sigmoid_high_lr'},
    ]
    results = {}
    val_fraction = CONFIG.get('val_split', 0.1)
    (_, _), (x_val, y_val) = train_val_split(
        x_train, y_train, val_fraction=val_fraction, seed=getattr(args, 'seed', 42))
    probe_X = x_val[:500]

    for cfg_ov in configs_to_run:
        tag      = cfg_ov.pop('tag')
        run_args = make_args(args, {
            **cfg_ov,
            'optimizer'       : 'sgd',
            'hidden_size'     : [128, 128, 128],
            'weight_decay'    : 0.0,
            'model_save_path' : f'dead_{tag}.npy',
        })
        sub   = _begin_sub_run(run_args, f'dead_{tag}', '2.5_dead_neuron',
                               {'experiment': '2.5_dead_neuron', 'tag': tag})
        model = NeuralNetwork(run_args)
        history = model.train(x_train, y_train,
                              epochs=run_args.epochs, batch_size=run_args.batch_size,
                              save_dir=run_args.save_dir, wandb_run=sub)

        val_accs = history.get('val_acc', [])
        if val_accs and sub is not None and _WANDB_AVAILABLE:
            try:
                sub.log({'plateau_epoch': int(np.argmax(val_accs)),
                         'best_val_acc' : float(np.max(val_accs))})
            except Exception:
                pass

        dead_fracs, grad_summaries, act_distributions = [], [], []
        current_input = probe_X
        for li, layer in enumerate(model.layers[:-1]):
            Z         = current_input @ layer.W + layer.b
            A         = layer.activation(Z)
            dead_mask = np.all(A == 0, axis=0)
            frac      = float(np.mean(dead_mask))
            dead_fracs.append(frac)
            print(f'  Layer {li}: {frac*100:.1f}% dead '
                  f'({int(dead_mask.sum())}/{len(dead_mask)})')
            act_distributions.append({
                'mean'        : float(A.mean()),
                'std'         : float(A.std()),
                'percent_zero': float((A == 0).mean() * 100),
                'raw'         : A.copy(),
            })
            gf = layer.gradient_flow_summary()
            grad_summaries.append(float(gf.mean()) if gf is not None else 0.0)
            if sub is not None and _WANDB_AVAILABLE:
                try:
                    sub.log({f'dead_frac/layer_{li}': frac})
                except Exception:
                    pass
            current_input = A

        results[tag] = {
            'model': model, 'dead_fracs': dead_fracs,
            'grad_summaries': grad_summaries,
            'act_distributions': act_distributions,
            'val_accs': val_accs, 'run_args': run_args,
        }
        _finish(sub)

    wandb_run = restore_outer(wandb_run)
    for fig, fname, key, cap in [
        (plot_dead_bar(results),   '2.5_dead_neuron_bar.png',
         '2.5/dead_neuron_bar',    'Dead neuron fraction per layer'),
        (plot_dead_dist(results, probe_X), '2.5_activation_dist.png',
         '2.5/activation_distribution',   'Activation distribution'),
        (plot_grad_comparison(results),    '2.5_gradient_flow.png',
         '2.5/gradient_flow',             'Gradient flow per layer'),
        (plot_val_accuracy(results),       '2.5_val_accuracy.png',
         '2.5/val_accuracy',              'Validation accuracy curves'),
    ]:
        save_and_log(fig, fname, args.save_dir, key, wandb_run, caption=cap)
        plt.close(fig)
    return wandb_run, results


# ── 2.6 loss comparison ──────────────────────────────────────────────────────

def loss_function_comparison(args, CONFIG, x_train, y_train,
                              NeuralNetwork=None, wandb_run=None):
    loss_types = ['cross_entropy', 'mse']
    history    = {}

    for loss_type in loss_types:
        run_args = make_args(args, {
            'loss'            : loss_type,
            'optimizer'       : 'rmsprop',
            'learning_rate'   : 0.0001,
            'hidden_size'     : [128, 128, 64],
            'activation'      : 'relu',
            'weight_decay'    : 0.0,
            'model_save_path' : f'loss_{loss_type}.npy',
        })
        sub   = _begin_sub_run(run_args, f'loss_{loss_type}', '2.6_loss_comparison',
                               {'experiment': '2.6', 'loss_type': loss_type})

        tl, vl, ta, va = [], [], [], []
        model         = NeuralNetwork(run_args)
        orig_evaluate = model.evaluate.__func__

        def _make_hook(tl_, vl_, ta_, va_):
            def hook(self_ref, X, y_oh, split_name='val'):
                m = orig_evaluate(self_ref, X, y_oh, split_name)
                if split_name == 'train':
                    tl_.append(m['loss']); ta_.append(m['accuracy'])
                elif split_name == 'val':
                    vl_.append(m['loss']); va_.append(m['accuracy'])
                return m
            return hook

        model.evaluate = types.MethodType(_make_hook(tl, vl, ta, va), model)
        model.train(x_train, y_train,
                    epochs=run_args.epochs, batch_size=run_args.batch_size,
                    save_dir=run_args.save_dir, wandb_run=sub)
        history[loss_type] = {
            'train_loss': tl, 'val_loss': vl,
            'train_acc' : ta, 'val_acc' : va,
        }
        _finish(sub)

    wandb_run = restore_outer(wandb_run)
    fig = plot_loss_comparison(history)
    save_and_log(fig, '2.6_loss_comparison.png', args.save_dir,
                 '2.6/loss_comparison', wandb_run,
                 caption='Loss & accuracy: CE vs MSE')
    plt.close(fig)
    return wandb_run, history


# ── 2.7 global performance overlay ───────────────────────────────────────────

def global_performance_overlay(run_records, save_dir='.', wandb_run=None):
    if not run_records:
        print('No run records — skipping 2.7 overlay.')
        return

    names      = [r['name']      for r in run_records]
    train_accs = np.array([r['train_acc'] for r in run_records])
    test_accs  = np.array([r['test_acc']  for r in run_records])
    gap        = train_accs - test_accs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    sc = ax.scatter(train_accs, test_accs, c=gap, cmap='RdYlGn_r',
                    s=60, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Train - Test gap')
    lim = [min(train_accs.min(), test_accs.min()) - 0.02,
           max(train_accs.max(), test_accs.max()) + 0.02]
    ax.plot(lim, lim, 'k--', linewidth=1, label='Perfect generalisation')
    for i in np.argsort(gap)[-5:]:
        ax.annotate(names[i], (train_accs[i], test_accs[i]),
                    fontsize=6, ha='right', color='darkred')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Train Accuracy'); ax.set_ylabel('Test Accuracy')
    ax.set_title('Train vs Test Accuracy (all runs)'); ax.legend(fontsize=8)

    axes[1].hist(gap, bins=20, color='#e74c3c', alpha=0.8,
                 edgecolor='k', linewidth=0.5)
    axes[1].axvline(0,    color='k',       linestyle='--', linewidth=1.2)
    axes[1].axvline(0.05, color='darkred', linestyle='--', linewidth=1.2,
                    label='Overfit threshold')
    axes[1].set_xlabel('Train - Test Gap'); axes[1].set_ylabel('Runs')
    axes[1].set_title('Gap Distribution'); axes[1].legend(fontsize=8)

    fig.suptitle('2.7 - Global Performance Overlay', fontsize=12)
    plt.tight_layout()
    save_and_log(fig, '2.7_global_performance_overlay.png', save_dir,
                 '2.7/global_performance_overlay', wandb_run)
    plt.close(fig)


def global_performance_overlay_from_wandb(args, wandb_run=None,
                                           api_key=None, project=None):
    if not _WANDB_AVAILABLE:
        print('wandb not available.')
        return
    api     = wandb.Api(api_key=api_key)
    project = project or getattr(args, 'wandb_project', 'DA6401_assignment1v6')
    entity  = getattr(args, 'wandb_entity', None)
    path    = f'{entity}/{project}' if entity else project
    records = []
    for r in api.runs(path):
        s  = r.summary._json_dict
        ta = s.get('train/accuracy')
        va = s.get('test/accuracy')
        if ta is not None and va is not None:
            records.append({'name': r.name,
                            'train_acc': float(ta), 'test_acc': float(va)})
    print(f'  Fetched {len(records)} runs for 2.7 overlay')
    global_performance_overlay(records, save_dir=args.save_dir,
                               wandb_run=wandb_run)


# ── 2.8 error analysis ───────────────────────────────────────────────────────

def error_analysis(model, x_test, y_test, dataset_name='mnist',
                   save_dir='.', wandb_run=None, class_names=None):
    from sklearn.metrics import confusion_matrix
    os.makedirs(save_dir, exist_ok=True)

    probs       = model.predict_proba(x_test)
    y_pred_lbl  = np.argmax(probs,  axis=1)
    y_true_lbl  = np.argmax(y_test, axis=1)
    confidences = probs.max(axis=1)
    cm          = confusion_matrix(y_true_lbl, y_pred_lbl)

    if class_names is None:
        class_names = (
            ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            if dataset_name.lower() in ('fashion_mnist', 'fashion')
            else [str(i) for i in range(10)]
        )

    fig_cm, ax = plt.subplots(figsize=(10, 8))
    # Use imshow for heatmap instead of sns.heatmap
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j],
                          ha='center', va='center', color='black', fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'2.8 - Confusion Matrix ({dataset_name})')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    save_and_log(fig_cm, '2.8_confusion_matrix.png', save_dir,
                 '2.8/confusion_matrix', wandb_run)
    plt.close(fig_cm)

    fig_c = creative_failure_viz(
        x_test, y_true_lbl, y_pred_lbl, probs, confidences, cm, class_names)
    save_and_log(fig_c, '2.8_creative_failure_analysis.png', save_dir,
                 '2.8/creative_failure_analysis', wandb_run)
    plt.close(fig_c)


# ── 2.9 weight init symmetry ─────────────────────────────────────────────────

def weight_init_symmetry(args, CONFIG, x_train, y_train,
                          NeuralNetwork=None, wandb_run=None,
                          n_neurons_to_track=5, track_grad_steps=50):
    inits   = ['zeros', 'xavier']
    history = {}

    for init in inits:
        run_args = make_args(args, {
            'weight_init'     : init,
            'optimizer'       : 'sgd',
            'learning_rate'   : 0.01,
            'hidden_size'     : [128, 128, 64],
            'activation'      : 'tanh',
            'weight_decay'    : 0.0,
            'epochs'          : 1,
            'model_save_path' : f'sym_{init}.npy',
        })
        sub   = _begin_sub_run(run_args, f'sym_{init}',
                               '2.9_weight_init_symmetry',
                               {'experiment': '2.9_symmetry', 'init': init})
        model = NeuralNetwork(run_args)
        model.grad_history_layer0 = []
        model.train(x_train, y_train,
                    epochs=run_args.epochs, batch_size=run_args.batch_size,
                    save_dir=run_args.save_dir, wandb_run=sub,
                    track_grad_steps=track_grad_steps)

        grad_arr = np.array(model.grad_history_layer0)
        if grad_arr.ndim < 2 or len(grad_arr) == 0:
            _finish(sub); continue

        history[init] = grad_arr
        if sub is not None and _WANDB_AVAILABLE:
            n_track = min(n_neurons_to_track, grad_arr.shape[1])
            for step_i, row in enumerate(grad_arr):
                log_d = {f'grad_neuron_{j}': float(row[j]) for j in range(n_track)}
                log_d['step'] = step_i
                try:
                    sub.log(log_d)
                except Exception:
                    pass
        _finish(sub)

    wandb_run = restore_outer(wandb_run)
    fig = plot_symmetry(history, n_neurons_to_track, track_grad_steps)
    save_and_log(fig, '2.9_weight_init_symmetry.png', args.save_dir,
                 '2.9/weight_init_symmetry', wandb_run)
    plt.close(fig)
    return wandb_run, history


# ── 2.10 fashion transfer ────────────────────────────────────────────────────

def fashion_mnist_transfer(args, CONFIG, x_train, y_train, x_test, y_test,
                            NeuralNetwork=None, wandb_run=None):
    BEST_3 = [
        {'name': 'Config1_momentum_relu_128_128_64',
         'optimizer': 'momentum', 'activation': 'relu',
         'hidden_size': [128, 128, 64], 'learning_rate': 0.01,
         'weight_decay': 0.0, 'batch_size': 64,
         'epochs': getattr(args, 'epochs', 30)},
        {'name': 'Config2_momentum_tanh_128_128_64',
         'optimizer': 'momentum', 'activation': 'tanh',
         'hidden_size': [128, 128, 64], 'learning_rate': 0.01,
         'weight_decay': 0.0, 'batch_size': 64,
         'epochs': getattr(args, 'epochs', 30)},
        {'name': 'Config3_rmsprop_relu_128_128_64',
         'optimizer': 'rmsprop', 'activation': 'relu',
         'hidden_size': [128, 128, 64], 'learning_rate': 0.0005,
         'weight_decay': 0.0, 'batch_size': 128,
         'epochs': getattr(args, 'epochs', 30)},
    ]
    results = []

    for cfg_def in BEST_3:
        name     = cfg_def['name']
        run_args = make_args(args, {
            **cfg_def,
            'dataset'         : 'fashion_mnist',
            'loss'            : 'cross_entropy',
            'weight_init'     : 'xavier',
            'model_save_path' : f'fashion_{name}.npy',
        })
        sub   = _begin_sub_run(run_args, f'fashion_{name}',
                               '2.10_fashion_transfer',
                               {'experiment': '2.10', 'config_name': name})
        model = NeuralNetwork(run_args)
        model.train(x_train, y_train,
                    epochs=run_args.epochs, batch_size=run_args.batch_size,
                    save_dir=run_args.save_dir, wandb_run=sub)
        test_m = model.evaluate(x_test, y_test, split_name='test')
        print(f'  TEST {name}: Acc={test_m["accuracy"]:.4f}  F1={test_m["f1"]:.4f}')
        if sub is not None and _WANDB_AVAILABLE:
            try:
                sub.log({'test/accuracy': test_m['accuracy'],
                         'test/f1'      : test_m['f1'],
                         'test/loss'    : test_m['loss']})
            except Exception:
                pass
        _finish(sub)
        results.append({'name': name, 'test_acc': test_m['accuracy'],
                        'test_f1': test_m['f1']})

    wandb_run = restore_outer(wandb_run)
    if wandb_run is not None and _WANDB_AVAILABLE:
        for r in results:
            try:
                wandb_run.log({f'2.10/test_acc/{r["name"]}': r['test_acc'],
                               f'2.10/test_f1/{r["name"]}' : r['test_f1']})
            except Exception:
                pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette   = ['#e74c3c', '#3498db', '#2ecc71']
    short     = [r['name'].split('_', 1)[1][:20] for r in results]
    for ax, metric, ylabel in zip(axes, ['test_acc', 'test_f1'],
                                  ['Test Accuracy', 'Test F1']):
        vals = [r[metric] for r in results]
        bars = ax.bar(short, vals, color=palette, alpha=0.85,
                      edgecolor='k', linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1); ax.set_ylabel(ylabel)
        ax.set_title(f'2.10 Fashion-MNIST: {ylabel}')
    plt.suptitle('2.10 - Fashion-MNIST Transfer Challenge', fontsize=11)
    plt.tight_layout()
    save_and_log(fig, '2.10_fashion_transfer_summary.png', args.save_dir,
                 '2.10/fashion_transfer_summary', wandb_run)
    plt.close(fig)
    return wandb_run, results