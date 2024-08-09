import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env import LEN_GAME_STATE

def hard_tanh(x):
    for i in len(x):
        if x[i].item() < -0.25:
            x[i] = -1

class RunnerModule(nn.Module):
    def __init__(self):
        super(RunnerModule, self).__init__()
        self.fc1 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE)  # Input layer
        self.fc2 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE//2)
        self.fc3 = nn.Linear(LEN_GAME_STATE//2, 2)    # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.tanh(self.fc3(x))
        return x
    
class ChaserModule(nn.Module):
    def __init__(self):
        super(ChaserModule, self).__init__()
        self.fc1 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE)  # Input layer
        self.fc2 = nn.Linear(LEN_GAME_STATE, LEN_GAME_STATE//2)
        self.fc3 = nn.Linear(LEN_GAME_STATE//2, 2)    # Output layer

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = torch.round(x)
        return x