"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np


def initialize_weights(weight_init, input_size, output_size):
    """Kept here for backward-compat imports."""
    if weight_init == 'zeros':
        return np.zeros((input_size, output_size))
    elif weight_init == 'random':
        return np.random.randn(input_size, output_size) * 0.01
    else:   # xavier
        std = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * std


def load_dataset(dataset: str):
    """
    Load dataset and return train / test splits.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)

    X shape : (N, 784)  float32 in [0, 1]
    y shape : (N, 10)   one-hot encoded

    The full Keras training set (60k) is returned as X_train.
    Validation split is handled internally by NeuralNetwork.train().
    Test set is the original Keras test split — never used during training.
    """
    dataset = dataset.lower().replace('-', '_')

    if dataset == 'mnist':
        from keras.datasets import mnist as _ds
        print("Loading MNIST ...")
    elif dataset in ('fashion_mnist', 'fashion'):
        from keras.datasets import fashion_mnist as _ds
        print("Loading Fashion-MNIST ...")
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion_mnist'.")

    (x_train, y_train), (x_test, y_test) = _ds.load_data()

    x_train = np.asarray(x_train, dtype='float32').reshape(-1, 784) / 255.0
    x_test  = np.asarray(x_test,  dtype='float32').reshape(-1, 784) / 255.0
    y_train = np.eye(10)[np.asarray(y_train)]
    y_test  = np.eye(10)[np.asarray(y_test)]

    print(f"  x_train:{x_train.shape}  x_test:{x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def train_val_split(X, y, val_fraction=0.1, seed=42):
    """Stratified train / validation split."""
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed,
        stratify=np.argmax(y, axis=1),
    )
    print(f"  x_train:{X_train.shape}  x_val:{X_val.shape}")
    return (X_train, y_train), (X_val, y_val)


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
    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1}