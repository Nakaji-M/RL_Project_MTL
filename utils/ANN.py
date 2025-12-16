import torch
from torch import nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(ANN, self).__init__()
        torch.manual_seed(seed)

        # Normalization layer
        self.norm = nn.LayerNorm(state_size)

        # Neural network layers
        self.fc1 = nn.Linear(state_size, 96)
        self.fc2 = nn.Linear(96, 192)
        self.fc3 = nn.Linear(192, action_size)

    def forward(self, state):
        # Normalize the input state
        x = self.norm(state)

        # Normal forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
