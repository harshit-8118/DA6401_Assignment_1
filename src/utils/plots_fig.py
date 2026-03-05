import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_dead_bar(results):
    palette  = ['#e74c3c', '#3498db', '#2ecc71']
    tags     = list(results.keys())
    n_layers = max(len(r['dead_fracs']) for r in results.values())
    width    = 0.25
    x_pos    = np.arange(n_layers)

    fig, ax  = plt.subplots(figsize=(9, 4))
    for ci, tag in enumerate(tags):
        fracs = results[tag]['dead_fracs']
        # pad if this run has fewer layers
        fracs = fracs + [0.0] * (n_layers - len(fracs))
        ax.bar(x_pos + ci * width, [f * 100 for f in fracs], width=width,
               color=palette[ci % len(palette)], alpha=0.85, label=tag,
               edgecolor='k', linewidth=0.6)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'Layer {i}' for i in range(n_layers)])
    ax.set_xlabel('Hidden Layer Index')
    ax.set_ylabel('Dead Neurons (%)')
    ax.set_title('2.5 - Dead Neuron Fraction per Layer')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_dead_dist(results, x_probe):
    palette = ['#e74c3c', '#3498db', '#2ecc71']   # fix: was only 2 colors
    probe_X = x_probe[:500]
    fig, ax = plt.subplots(figsize=(8, 4))

    for ci, (tag, res) in enumerate(results.items()):
        model = res['model']
        Z     = probe_X @ model.layers[0].W + model.layers[0].b
        A     = model.layers[0].activation(Z)
        ax.hist(A.flatten(), bins=60, alpha=0.55,
                color=palette[ci % len(palette)], label=tag,
                density=True, edgecolor='black', linewidth=0.5)

    ax.axvline(0, color='k', linestyle='--', linewidth=1.2, label='zero')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')
    ax.set_title('2.5 - Activation Distribution (Hidden Layer 0)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_grad_comparison(results):
    palette  = ['#e74c3c', '#3498db', '#2ecc71']
    tags     = list(results.keys())
    n_layers = max(len(r['grad_summaries']) for r in results.values())
    width    = 0.25
    x_pos    = np.arange(n_layers)

    fig, ax  = plt.subplots(figsize=(9, 4))
    for ci, tag in enumerate(tags):
        grads = results[tag]['grad_summaries']
        grads = grads + [0.0] * (n_layers - len(grads))
        ax.bar(x_pos + ci * width, grads, width=width,
               color=palette[ci % len(palette)], alpha=0.85, label=tag,
               edgecolor='k', linewidth=0.6)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'Layer {i}' for i in range(n_layers)])
    ax.set_xlabel('Hidden Layer Index')
    ax.set_ylabel('Mean Gradient Magnitude')
    ax.set_title('2.5 - Gradient Flow per Layer (ReLU vs Tanh vs Sigmoid)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_val_accuracy(results):
    palette = ['#e74c3c', '#3498db', '#2ecc71']
    fig, ax = plt.subplots(figsize=(9, 4))

    for ci, (tag, res) in enumerate(results.items()):
        val_accs = res.get('val_accs', [])
        if not val_accs:
            print(f"  Warning: no val_accs found for {tag}, skipping.")
            continue
        epochs = np.arange(1, len(val_accs) + 1)
        ax.plot(epochs, val_accs, color=palette[ci % len(palette)],
                label=tag, linewidth=2)
        # mark plateau point
        plateau_epoch = int(np.argmax(val_accs))
        ax.axvline(plateau_epoch + 1, color=palette[ci % len(palette)],
                   linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('2.5 - Validation Accuracy (Plateau Detection)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_loss_comparison(history):
    colors = {'cross_entropy': '#e74c3c', 'mse': '#3498db'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for loss_type, h in history.items():
        c      = colors[loss_type]
        epochs = range(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs, h['train_loss'], color=c, ls='-',
                     label=f'{loss_type} train', linewidth=1.8)
        axes[0].plot(epochs, h['val_loss'],   color=c, ls='--',
                     label=f'{loss_type} val',   linewidth=1.5)
        axes[1].plot(epochs, h['train_acc'],  color=c, ls='-',
                     label=f'{loss_type} train', linewidth=1.8)
        axes[1].plot(epochs, h['val_acc'],    color=c, ls='--',
                     label=f'{loss_type} val',   linewidth=1.5)
    axes[0].set_title('2.6 - Loss Convergence')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[1].set_title('2.6 - Accuracy Convergence')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    return fig

def creative_failure_viz(X, y_true, y_pred, probs, confidences, cm, class_names):
    n_classes = len(class_names)
    fig       = plt.figure(figsize=(16, 5))
    gs        = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_a = fig.add_subplot(gs[0])
    per_class_acc = []
    for c in range(n_classes):
        mask = (y_true == c)
        per_class_acc.append(np.mean(y_pred[mask] == c) if mask.sum() > 0 else 0.0)
    sorted_pairs = sorted(zip(per_class_acc, class_names))
    sorted_acc   = [p[0] for p in sorted_pairs]
    sorted_names = [p[1] for p in sorted_pairs]
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]
    bars = ax_a.barh(sorted_names, sorted_acc,
                     color=colors)
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel('Accuracy')
    ax_a.set_title('Per-class Accuracy')
    for bar, acc in zip(bars, sorted_acc):
        ax_a.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                  f'{acc:.2f}', va='center', fontsize=7)

    ax_b   = fig.add_subplot(gs[1])
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    i_true, i_pred = np.unravel_index(cm_off.argmax(), cm_off.shape)
    confused_idx   = np.where((y_true == i_true) & (y_pred == i_pred))[0]
    np.random.shuffle(confused_idx)
    n_show = min(8, len(confused_idx))
    if n_show > 0:
        cols     = 4
        rows     = (n_show + cols - 1) // cols
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=gs[1], hspace=0.05, wspace=0.05)
        ax_b.set_visible(False)
        for k in range(n_show):
            ax_img = fig.add_subplot(inner_gs[k // cols, k % cols])
            ax_img.imshow(X[confused_idx[k]].reshape(28, 28), cmap='gray')
            ax_img.set_title(f'{probs[confused_idx[k], i_pred]:.2f}', fontsize=6)
            ax_img.axis('off')
        fig.text(0.51, 0.95,
                 f'Most confused: True={class_names[i_true]}  ->  '
                 f'Pred={class_names[i_pred]}  ({cm_off[i_true, i_pred]} cases)',
                 ha='center', fontsize=9, fontweight='bold')
    else:
        ax_b.text(0.5, 0.5, 'No confused samples', ha='center', va='center')

    ax_c      = fig.add_subplot(gs[2])
    correct_c = confidences[y_pred == y_true]
    wrong_c   = confidences[y_pred != y_true]
    bins      = np.linspace(0, 1, 30)
    ax_c.hist(correct_c, bins=bins, alpha=0.65, color='#2ecc71',
              label=f'Correct (n={len(correct_c)})', density=True, edgecolor='black', linewidth=0.5)
    ax_c.hist(wrong_c,   bins=bins, alpha=0.65, color='#e74c3c',
              label=f'Wrong  (n={len(wrong_c)})',   density=True, edgecolor='black', linewidth=0.5)
    ax_c.set_xlabel('Predicted Confidence')
    ax_c.set_ylabel('Density')
    ax_c.set_title('Confidence: Correct vs Wrong')
    ax_c.legend(fontsize=8)
    fig.suptitle('2.8 - Creative Failure Analysis', fontsize=12, y=1.01)
    return fig

def plot_symmetry(history, n_neurons_to_track=5, max_steps=50):
    cmap = plt.cm.tab10
    palette = [cmap(i % 10) for i in range(n_neurons_to_track)]
    titles  = {
        'zeros' : 'Zeros Init — all neurons identical (symmetry NOT broken)',
        'xavier': 'Xavier Init — neurons diverge (symmetry broken)',
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, init in zip(axes, ['zeros', 'xavier']):
        grad_arr = history.get(init, np.zeros((max_steps, n_neurons_to_track)))
        steps    = np.arange(1, len(grad_arr) + 1)
        n_track  = min(n_neurons_to_track, grad_arr.shape[1])

        for j in range(n_track):
            ax.plot(steps, grad_arr[:, j], color=palette[j],
                    linewidth=1.9, label=f'Neuron {j}', alpha=0.85)

        # fix: annotate overlap for zeros
        if init == 'zeros':
            ax.text(0.5, 0.5, 'All neurons overlap\n(identical gradients)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='red', alpha=0.6,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('SGD Update Step')
        ax.set_ylabel('Gradient Norm (layer 0)')
        ax.set_title(titles.get(init, init), fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('2.9 - Weight Initialisation & Symmetry Breaking', fontsize=11)
    plt.tight_layout()
    return fig
