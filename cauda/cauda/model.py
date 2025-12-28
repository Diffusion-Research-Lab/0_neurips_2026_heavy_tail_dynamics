"""Neural network model for the vector field in the diffusion process."""

# Authors: Hamza Cherkaoui

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class TimeCondBlock(nn.Module):
    def __init__(self, width: int = 64, tdim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.tproj = nn.Linear(tdim, width)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        skip = h
        h = self.fc1(h)
        h = self.act(h)
        h = h + self.tproj(temb)
        h = self.act(h)
        h = self.fc2(h)
        return h + skip


class LightNet(nn.Module):
    def __init__(self, dim: int = 2, width: int = 64, n_blocks: int = 4, tdim: int = 32):
        super().__init__()
        self.temb = TimeEmbedding(hidden=tdim)
        self.inp = nn.Linear(dim, width)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList([TimeCondBlock(width=width, tdim=tdim) for _ in range(n_blocks)])
        self.out = nn.Linear(width, dim)

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        temb = self.temb(t_norm)
        h = self.act(self.inp(x))
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(h)
