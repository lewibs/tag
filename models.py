import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env import LEN_GAME_STATE

class RunnerModule(nn.Module):
    def __init__(self):
        super(RunnerModule, self).__init__()
        self.fc1 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE)  # Input layer
        self.fc2 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE//2)
        self.fc3 = nn.Linear(LEN_GAME_STATE//2, 2)    # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.clamp(x, 0, 1)
        return x
    
class ChaserModule(nn.Module):
    def __init__(self):
        super(ChaserModule, self).__init__()
        self.fc1 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE)  # Input layer
        self.fc2 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE//2)
        self.fc3 = nn.Linear(LEN_GAME_STATE//2, 2)    # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.clamp(x, 0, 1)
        return x