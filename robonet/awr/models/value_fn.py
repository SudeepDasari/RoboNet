import torch.nn as nn
import torch.nn.functional as F
import torch


class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()
        self._fc1 = nn.Linear(128, 64)
        self._fc2 = nn.Linear(64, 1)
    
    def forward(self, state):
        fc1 = F.relu(self._fc1(state))
        return F.relu(self._fc2(fc1))
