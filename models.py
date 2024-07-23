import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RunnerModule(nn.Module):
    def __init__(self):
        super(RunnerModule, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)   # Hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x