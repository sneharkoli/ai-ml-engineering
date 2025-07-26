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

# Add this line for adaptive learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

best_accuracy = 0.0
epochs_no_improve = 0
patience = 2  # Stop if no improvement for 2 consecutive epochs

# 5. Training loop
for epoch in range(100):  # You can increase this number now if you want
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

    # Step the scheduler
    scheduler.step(accuracy)

    # Save the model if accuracy improves
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