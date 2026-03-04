## Roll no: DA25S003, Harshit Shukla

# Assignment 1: Multi-Layer Perceptron for Image Classification

## WANDB_REPORT
[https://api.wandb.ai/links/da25s003-indian-institute-of-technology-madras/3tfhxatv](https://api.wandb.ai/links/da25s003-indian-institute-of-technology-madras/3tfhxatv)

## GITHUB_LINK
[https://github.com/harshit-8118/DA6401_Assignment_1](https://github.com/harshit-8118/DA6401_Assignment_1)


A fully implemented neural network in NumPy with support for multiple optimizers, activations, and experiments.

---

## Folder Structure

```
Assignment-1
│
├── src/
│    ├── ann/                          # Core neural network
│    │   ├── neural_network.py         # Main NeuralNetwork class
│    │   ├── neural_layer.py           # Layer implementation (forward/backward)
│    │   ├── activations.py            # ReLU, Tanh, Sigmoid, Softmax
│    │   ├── optimizers.py             # SGD, Momentum, NAG, RMSProp
│    │   └── objective_functions.py    # MSE, Cross-Entropy loss
│    │   └── __init__.py               # Exports all utilities
│    ├── utils/
│    │   ├── arguments.py              # CLI argument parser & CONFIG
│    │   ├── data_loader.py            # MNIST & Fashion-MNIST Loading | Scaling | Splitting | Initializing weights
│    │   ├── plots_fig.py              # Visualization functions
│    │   ├── wandb_report.py           # W&B logging
│    │   └── __init__.py               # Exports all utilities
│    ├── train.py                      # Main training script
│    ├── inference.py                  # Model testing script
│    ├── best_model.npy               # Saved weights
│    └── best_model_config.json       # Model configuration
└── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy
- matplotlib, seaborn (plotting)
- keras (dataset loading) & tensorflow for bringing keras in action
- wandb (optional, for experiment tracking)
- scikit-learn (metrics)

---

## Quick Start

### Train Model (MNIST)
```bash
python src/train.py --dataset mnist --epochs 50 --batch_size 32 \
  --optimizer momentum --learning_rate 0.01 \
  --activation relu --num_neurons 128 128 64 \
  --weight_init xavier --experiment train
```

### Train Model (Fashion-MNIST)
```bash
python src/train.py --dataset fashion_mnist --epochs 50 --batch_size 64 \
  --optimizer momentum --activation tanh --experiment train
```

### Run Inference
```bash
python src/inference.py --no_wandb --model_save_path best_model.npy \
 --save_dir src/ --dataset mnist
```

### Run All Experiments (Requires W&B)

```bash
python src/train.py --experiment all --dataset mnist
```

---

## Core Classes & Functions

### NeuralNetwork (`ann/neural_network.py`)

**Constructor:**
```python
model = NeuralNetwork(CONFIG, cli_args)
```
- `CONFIG`: Dict with `val_split`, `beta` (momentum), `epsilon` (RMSProp)
- `cli_args`: Arguments from command line (learning_rate, optimizer, activation, etc.)
- Builds layers automatically from `num_neurons` argument
- Initializes optimizer based on selection

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `forward(X)` | Forward pass, returns logits (no softmax) |
| `predict_proba(X)` | Returns softmax probabilities |
| `predict(X)` | Returns class labels |
| `backward(y_true, y_pred)` | Backpropagation, returns gradients |
| `update_weights()` | Updates weights using optimizer |
| `train(X, y, epochs, batch_size, save_dir, wandb_run)` | Full training loop with early stopping on validation F1 |
| `evaluate(X, y, split_name)` | Returns loss, accuracy, precision, recall, F1 |

**Save/Load:**
```python
model.save_model(save_dir)  # Saves weights (.npy) + config (.json)

model = NeuralNetwork.load(weights_path, config_path, CONFIG)
```

---

### NeuralLayer (`ann/neural_layer.py`)

**Constructor:**
```python
layer = NeuralLayer(input_size=784, output_size=128, 
                    activation='relu', weight_init='xavier')
```

**Initialization Methods:**
- `'random'`: `W ~ N(0, 0.01)`
- `'xavier'`: `W ~ N(0, sqrt(2/(in+out)))`
- `'zeros'`: `W = 0`

**Forward/Backward:**
```python
A = layer.forward(X)           # Output after activation
delta = layer.backward(delta)  # Backprop, returns delta for previous layer
```

**Utilities:**
- `dead_neuron_fraction(X)`: % neurons outputting zero (ReLU only)
- `activation_distribution(X)`: Mean, std, min, max, zero %, raw activations
- `gradient_flow_summary()`: Average gradient magnitude per neuron

---

### Data Loading (`utils/data_loader.py`)

```python
(X_train, y_train), (X_test, y_test) = load_dataset('mnist')
# or 'fashion_mnist'
```

**Preprocessing:**
- Flattens 28×28 images → 784 features
- Normalizes pixels to [0, 1]
- Converts labels to one-hot (10 classes)

**Utilities:**
```python
compute_metrics(y_true_labels, y_pred_labels)  # Returns accuracy, precision, recall, F1

train_val_split(X, y, val_fraction=0.2, seed=42)  # 80-20 split with stratification
```

---

## Training Details

### train.py Workflow

1. **Parse arguments** → Load dataset → Create W&B run (if enabled)
2. **Run experiment** based on `--experiment` flag:
   - `train`: Train best model, save weights + config
   - `visual`: Display 5 samples per class
   - `sweep`: Hyperparameter sweep (100+ runs)
   - `optimizer`: Compare SGD vs Momentum vs NAG vs RMSProp
   - `vanishing`: Analyze gradient vanishing
   - `dead`: Track dead neurons per layer
   - `loss`: Compare MSE vs Cross-Entropy
   - `error`: Analyze misclassified samples
   - `fashion`: Transfer learning MNIST→Fashion-MNIST
   - `all`: Run all experiments

3. **Best model saved automatically** when validation F1 improves
   - Weights: `models/best_model.npy`
   - Config: `models/best_model_config.json`

### Training Loop (train.py + NeuralNetwork.train())

```
For each epoch:
  Shuffle training data
  For each batch:
    Forward pass → Backward pass → Gradient descent
    (NAG includes lookahead step)
  
  Evaluate on train & validation sets
  Log metrics to W&B
  
  If val_f1 > best_val_f1:
    Save model weights + config
```

---

## Inference Details

### inference.py Workflow

1. **Load model** from `best_model.npy` + `best_model_config.json`
2. **Batch prediction** (process test set in chunks)
   - Forward pass for logits
   - Apply softmax → get probabilities
   - Argmax → predicted class
3. **Compute metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
4. **Save results** to `models/inference_results.json`
5. **Log to W&B** (if enabled)

---

## Command-Line Arguments

| Argument | Default | Options |
|----------|---------|---------|
| `-d, --dataset` | mnist | mnist, fashion_mnist, fashion |
| `-e, --epochs` | 50 | int |
| `-b, --batch_size` | 32 | int |
| `-lr, --learning_rate` | 0.01 | float |
| `-o, --optimizer` | momentum | sgd, momentum, nag, rmsprop |
| `-a, --activation` | relu | relu, tanh, sigmoid |
| `-l, --loss` | cross_entropy | mse, cross_entropy |
| `-sz, --num_neurons` | 128 128 64 | list of ints |
| `-wi, --weight_init` | xavier | random, xavier, zeros |
| `-wd, --weight_decay` | 0.0 | float |
| `--experiment` | train | train, visual, sweep, optimizer, ... |
| `--save_dir` | models | str (folder path) |
| `--no_wandb` | False | (skip W&B logging) |
| `--seed` | 42 | int |

---

## Example Workflows

### Scenario 1: Train & Test on MNIST
```bash
# Train
python src/train.py --dataset mnist --epochs 50 --optimizer momentum --experiment train

# Test
python src/inference.py --dataset mnist
```

### Scenario 2: Fashion-MNIST with Tanh
```bash
python src/train.py --dataset fashion_mnist --activation tanh \
  --batch_size 64 --learning_rate 0.01 --experiment train

python src/inference.py --dataset fashion_mnist
```

### Scenario 3: Compare Optimizers
```bash
python src/train.py --dataset mnist --experiment optimizer
# Automatically trains SGD, Momentum, NAG, RMSProp and logs results to W&B
```

### Scenario 4: Hyperparameter Sweep (100+ runs)
```bash
python src/train.py --dataset mnist --experiment sweep
# Uses W&B Sweeps to test combinations of optimizers, activations, LR, etc.
# View parallel coordinates plot in W&B dashboard
```

---

## Key Implementation Notes

### Gradient Clipping
Gradients are clipped to ±5 to prevent explosion.

### Early Stopping
Saves model when validation F1 improves (best epoch automatically selected).

### NAG (Nesterov Accelerated Gradient)
Includes lookahead + restore steps in training loop.

### Weight Initialization Impact
- `random`: Often leads to vanishing gradients
- `xavier`: Balances scale relative to input/output size (recommended)
- `zeros`: Only for bias initialization

---

## What's NOT in This README

- **wandb_report.py**: Contains W&B logging & visualization functions (too many for brevity)
- **plots_fig.py**: Plotting utilities for experiments
- **activations.py**: Just activation functions (ReLU, Tanh, Sigmoid, Softmax)
- **optimizers.py**: Just optimizer update rules (see parameter descriptions above)
- **objective_functions.py**: Just loss functions (MSE, Cross-Entropy)

These are straightforward and well-documented in their respective files.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `best_model.npy` not found | Run `--experiment train` first |
| W&B errors | Use `--no_wandb` flag |
| Low validation accuracy | Try longer training, adjust learning_rate, check batch_size |
| Dead neurons | Switch to Tanh or lower learning rate |
| Gradient vanishing | Use Xavier initialization + ReLU |

---

## Summary

- **NeuralNetwork**: Main training/inference class
- **NeuralLayer**: Individual layer with forward/backward
- **train.py**: 10 different experiments + model training
- **inference.py**: Load & evaluate on test set
- **Key hyperparameters**: optimizer, learning_rate, activation, num_neurons, weight_init
- **Save/Load**: Automatic via `.save_model()` and `NeuralNetwork.load()`