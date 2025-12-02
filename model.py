"""Neural network model for the vector field in the diffusion process."""

# Authors: Hamza Cherkaoui

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Neural network to model the vector field in the diffusion process."""

    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=-1))