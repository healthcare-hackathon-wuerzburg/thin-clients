import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    """
    Simple neural network model with a flattened input, two fully connected layers, and a sigmoid activation.

    Attributes:
        flatten (torch.nn.Flatten): Flattens the input tensor.
        fc1 (torch.nn.Linear): First fully connected layer with ReLU activation.
        relu (torch.nn.ReLU): Rectified Linear Unit activation function.
        fc2 (torch.nn.Linear): Second fully connected layer with a sigmoid activation for binary classification.

    Methods:
        forward(x): Forward pass method to compute the output tensor given the input tensor.
    """

    def __init__(self):
        """
        Initialize the SimpleModel.

        Defines the layers of the neural network.
        """
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 512)  # Adjust the input size based on your image dimensions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4)  # Output size is 4 for the vector with 4 values

    def forward(self, x):
        """
        Forward pass method to compute the output tensor given the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with sigmoid activation.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)