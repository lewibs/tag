import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def format_positions(nearest, watching):
    while len(nearest) < watching:
        nearest.append([0, 0, 100000000])

    # Flatten the list of [angle, distance] pairs
    flattened_objects = [item for sublist in nearest for item in sublist]

    # Convert to a PyTorch tensor
    return torch.tensor([flattened_objects], dtype=torch.float32)

class RunnerModule(nn.Module):
    def __init__(self, watching):
        super(RunnerModule, self).__init__()
        self.fc1 = nn.Linear(watching*3, watching)  # Input layer
        self.fc2 = nn.Linear(watching, watching//2)
        self.fc3 = nn.Linear(watching//2, 2)    # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # Apply sign function and thresholding
        x = torch.sign(x)
        x[x.abs() < 0.5] = 0  # Set small values to 0
        
        return x
    
class ChaserModule(nn.Module):
    def __init__(self, watching):
        super(ChaserModule, self).__init__()
        self.fc1 = nn.Linear(watching*3, watching)  # Input layer
        self.fc2 = nn.Linear(watching, watching//2)
        self.fc3 = nn.Linear(watching//2, 2)    # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # Apply sign function and thresholding
        x = torch.sign(x)
        x[x.abs() < 0.5] = 0  # Set small values to 0
        
        return x