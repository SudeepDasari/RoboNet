import torch.nn as nn
import torch.nn.functional as F
import torch


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self._fc1 = nn.Linear(128, 64)
        self._fc2 = nn.Linear(64, 8)
    
    def forward(self, state):
        fc1 = F.relu(self._fc1(state))
        fc2 = F.relu(self._fc2(fc1))
        pred_mean = fc2[:,:4]
        pred_std = fc2[4:]
        return pred_mean, pred_std
