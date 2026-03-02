"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np 

#  Metrics 
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


def initialize_weights(weight_init, input_size, output_size):
    if weight_init == 'random':
        return np.random.randn(input_size, output_size) * 0.01
    else: 
        std = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * std


def train_val_split(X, y, val_fraction=0.2, seed=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )

    print(f"x_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"x_val: {X_val.shape} | y_val: {y_val.shape}")
    return (X_train, y_train), (X_val, y_val)


def load_dataset(dataset):
    if dataset == 'mnist':
         print("mnist data loading...")
         (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else: 
         print("fashion mnist data loading...")
         (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test  = np.asarray(x_test)
    y_test  = np.asarray(y_test)
    y_train = np.eye(10)[y_train]
    y_test  = np.eye(10)[y_test]
    x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
    x_test  = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

    print(f"x_train: {x_train.shape} | y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape} | y_test: {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)