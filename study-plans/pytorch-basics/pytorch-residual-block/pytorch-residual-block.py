import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Returns: None
        """
        super().__init__()
        self.layer1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Returns: output tensor
        """
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu(x + out)   # skip connection, then final activation
        return out