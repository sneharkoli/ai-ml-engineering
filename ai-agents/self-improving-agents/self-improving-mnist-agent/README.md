Here’s a **complete, beginner-friendly documentation** of your self-improving MNIST agent project, including all code and detailed explanations for each step.

---

# Self-Improving MNIST AI Agent

This project demonstrates how to build a simple neural network that learns to classify handwritten digits from the MNIST dataset, and **improves itself** using:
- Model checkpointing (saving the best model)
- Adaptive learning rate (automatically lowering the learning rate if learning stalls)
- Early stopping (stopping training if no improvement is seen)

---

## 1. **Loading and Exploring the MNIST Data**

**File:** `src/data/mnist_loader.py`

This script downloads the MNIST dataset, loads it into NumPy arrays, and provides a function to visualize the data.

```python
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
        buf = f.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows * cols) / 255.0  # Normalize to [0,1]
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
```

**Explanation:**
- Downloads MNIST data if not present.
- Loads images and labels into NumPy arrays.
- Normalizes images to [0, 1].
- Visualizes a few samples if run directly.

---

## 2. **Defining the Neural Network**

**File:** `src/models/simple_nn.py`

This file defines a simple feedforward neural network using PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 64)     # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)      # Hidden layer to output layer (10 classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

**Explanation:**
- Input: 784 features (28x28 pixels).
- Two hidden layers (128 and 64 neurons).
- Output: 10 neurons (one for each digit).
- Uses ReLU activation for hidden layers.

---

## 3. **Training the Self-Improving Agent**

**File:** train_simple_nn.py

This script trains the neural network and implements self-improvement strategies.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from models.simple_nn import SimpleNN
from data.mnist_loader import load_mnist

# 1. Load data
X_train, y_train, X_test, y_test = load_mnist()

# 2. Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 3. Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 4. Initialize model, loss, optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adaptive learning rate: Reduce LR if accuracy plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

best_accuracy = 0.0
epochs_no_improve = 0
patience = 2  # Stop if no improvement for 2 consecutive epochs

# 5. Training loop
for epoch in range(5):  # You can increase this number now if you want
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

    # Evaluate after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

    # Step the scheduler (adaptive learning rate)
    scheduler.step(accuracy)

    # Save the model if accuracy improves (checkpointing)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_simple_nn.pth")
        print("Best model saved!")
        epochs_no_improve = 0  # Reset counter if improved
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    # Early stopping
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

print(f"Best Test Accuracy: {best_accuracy:.2f}%")
```

**Explanation:**
- Loads and prepares MNIST data for PyTorch.
- Trains the neural network in batches.
- **Model checkpointing:** Saves the model whenever test accuracy improves.
- **Adaptive learning rate:** Reduces learning rate if accuracy plateaus.
- **Early stopping:** Stops training if no improvement for `patience` epochs.

---

## 4. **How the Self-Improvement Works**

- **Model Checkpointing:**  
  Keeps the best model found during training, so you never lose your best result.

- **Adaptive Learning Rate:**  
  If the model stops improving, the learning rate is reduced, allowing finer adjustments to weights and potentially escaping plateaus.

- **Early Stopping:**  
  If the model doesn't improve for a set number of epochs, training stops automatically, saving time and preventing overfitting.

---

## 5. **How to Run**

1. **Install dependencies:**
   ```sh
   pip install torch torchvision matplotlib numpy
   ```

2. **Train the agent:**
   ```sh
   python src/train_simple_nn.py
   ```

3. **Check your best model:**  
   The best model will be saved as `best_simple_nn.pth` in your project directory.

---

## 6. **Next Steps & Ideas**

- Try different neural network architectures (e.g., add convolutional layers).
- Experiment with hyperparameters (learning rate, batch size, patience).
- Add more advanced self-improvement strategies (active learning, pseudo-labeling).
- Visualize training curves (loss and accuracy over epochs).

---

**Results!**  
python src\train_simple_nn.py
Epoch 1 complete. Loss: 0.3423
Test Accuracy after epoch 1: 83.92%
Best model saved!
Epoch 2 complete. Loss: 0.5616
Test Accuracy after epoch 2: 85.81%
Best model saved!
Epoch 3 complete. Loss: 0.5554
Test Accuracy after epoch 3: 86.51%
Best model saved!
Epoch 4 complete. Loss: 0.3236
Test Accuracy after epoch 4: 86.26%
No improvement for 1 epoch(s).
Epoch 5 complete. Loss: 0.3829
Test Accuracy after epoch 5: 86.57%
Best model saved!
Epoch 6 complete. Loss: 0.1822
Test Accuracy after epoch 6: 87.05%
Best model saved!
Epoch 7 complete. Loss: 0.2068
Test Accuracy after epoch 7: 87.47%
Best model saved!
Epoch 8 complete. Loss: 0.1400
Test Accuracy after epoch 8: 88.08%
Best model saved!
Epoch 9 complete. Loss: 0.0552
Test Accuracy after epoch 9: 87.85%
No improvement for 1 epoch(s).
Epoch 10 complete. Loss: 0.2644
Test Accuracy after epoch 10: 88.42%
Best model saved!
Epoch 11 complete. Loss: 0.1560
Test Accuracy after epoch 11: 88.33%
No improvement for 1 epoch(s).
Epoch 12 complete. Loss: 0.2519
Test Accuracy after epoch 12: 87.93%
No improvement for 2 epoch(s).
Early stopping triggered.
Best Test Accuracy: 88.42%