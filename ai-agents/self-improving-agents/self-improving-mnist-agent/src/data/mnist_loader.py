import os
import gzip
import urllib.request
import numpy as np

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "mnist_data")

def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in MNIST_URLS.items():
        out_path = os.path.join(DATA_DIR, url.split('/')[-1])
        if not os.path.exists(out_path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, out_path)

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows * cols) / 255.0
        return data

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def load_mnist():
    download_mnist()
    paths = {k: os.path.join(DATA_DIR, v.split('/')[-1]) for k, v in MNIST_URLS.items()}
    X_train = load_images(paths["train_images"])
    y_train = load_labels(paths["train_labels"])
    X_test = load_images(paths["test_images"])
    y_test = load_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # Visualize a few training images
    import matplotlib.pyplot as plt

    for i in range(5):
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
        plt.show()