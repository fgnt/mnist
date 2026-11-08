import gzip
import struct
import os
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(filepath: str) -> np.ndarray:
    """
    Load MNIST images from a .gz file.

    Parameters:
    - filepath (str): Path to the .gz file containing MNIST images.

    Returns:
    - np.ndarray: Image array of shape (num_samples, 784) with values in range [0, 255].
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number, number of images, rows, and columns (big-endian 32-bit integers)
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for MNIST images file.")
        
        # Read the raw byte buffer and convert to uint8 array
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        
        # Reshape into (N, 784)
        return images.reshape(num_images, rows * cols)

def load_mnist_labels(filepath: str) -> np.ndarray:
    """
    Load MNIST labels from a .gz file.

    Parameters:
    - filepath (str): Path to the .gz file containing MNIST labels.

    Returns:
    - np.ndarray: Label array of shape (num_samples,) with values in range [0, 9].
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and number of items (big-endian 32-bit integers)
        magic, num_items = struct.unpack('>II', f.read(8))
        
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for MNIST labels file.")
        
        # Read the raw byte buffer and convert to uint8 array
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8) 
        return labels

def load_mnist_data(data_dir: str = "."):
    """
    Load both training and testing datasets for MNIST.

    Parameters:
    - data_dir (str): Root directory containing the four .gz MNIST files.

    Returns:
    - tuple: (X_train, y_train, X_test, y_test)
    """
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    return X_train, y_train, X_test, y_test

# Assuming you have the MNIST dataset files in the current directory, you can load them as follows:
X_train, y_train, X_test, y_test = load_mnist_data(data_dir=".")
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

def accuracy_score(y_true, y_pred):
    # Calculate the accuracy score as the ratio of correctly predicted samples to total samples
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def precision_score(y_true, y_pred, average: str = "macro"):
    """
    Calculate the precision score as the ratio of true positives to the sum of true positives and false positives
    - "macro" average: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    - None / binary: Calculate precision for the positive class (label 1) only. This is the default behavior.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if average == "macro" else [1]

    precision = []
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        # Handle zero division by returning 0 if there are no positive predictions
        precision.append(true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0)
    return np.mean(precision) if average == "macro" else precision[0]

def recall_score(y_true, y_pred, average: str = "macro"):
    """
    Calculate the recall score as the ratio of true positives to the sum of true positives and false negatives
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if average == "macro" else [1]

    recall = []
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))
        # Handle zero division by returning 0 if there are no positive instances
        recall.append(true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0)
    return np.mean(recall) if average == "macro" else recall[0]

def f1_score(y_true, y_pred, average: str = "macro"):
    """
    Calculate the F1 score as the harmonic mean of precision and recall
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if average == "macro" else [1]

    f1_scores = []
    for cls in classes:
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1_scores.append(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

    return np.mean(f1_scores) if average == "macro" else f1_scores[0]

def classification_report(y_true, y_pred):
    # Generate a classification report containing precision, recall, and F1 score for each class
    classes = np.unique(y_true)
    report = {}
    for cls in classes:
        report[cls] = {
            "precision": precision_score(y_true, y_pred, average=None),
            "recall": recall_score(y_true, y_pred, average=None),
            "f1-score": f1_score(y_true, y_pred, average=None)
        }
    return report

def roc_auc_curve(y_true, y_pred):
    return

class NaiveBayes():
    def __init__(self):
        self.class_priors = None
        self.likelihoods = None
        self.classes = None

    def predict():
        return

plt.legend()
plt.show()