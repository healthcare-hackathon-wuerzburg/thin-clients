import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 512)  # Adjust the input size based on your image dimensions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4)  # Output size is 4 for the vector with 4 values

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
