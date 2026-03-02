"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

from keras.datasets import mnist, fashion_mnist
import numpy as np 

def initialize_weights(weight_init, input_size, output_size):
    if weight_init == 'random':
        return np.random.randn(input_size, output_size) * 0.01
    else: 
        std = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * std


def train_val_split(X, y, val_fraction=0.2, seed=42):
    """
    Split data into training and validation sets.
    Ensures proper random splitting (no data leakage).
    """
    from sklearn.model_selection import train_test_split
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