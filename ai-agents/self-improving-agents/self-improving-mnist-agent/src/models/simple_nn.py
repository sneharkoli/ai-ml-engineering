import torch
import torch.nn as nn
import torch.nn.functional as F


"""
What’s Happening Here?
Input: Each MNIST image is 28x28 pixels, so input size is 784.
Layers: Two hidden layers (128 and 64 neurons), output layer has 10 neurons (digits 0-9).
Activation: ReLU for hidden layers, raw output for final (we’ll use softmax during training).
"""

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