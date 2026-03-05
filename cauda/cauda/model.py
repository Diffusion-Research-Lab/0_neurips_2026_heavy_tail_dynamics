"""Neural network model for the vector field in the diffusion process."""

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class LightNet(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        width: int = 32,
        n_blocks: int = 2,
        tdim: int = 16,
        bottleneck: int | None = None,
    ):
        super().__init__()
        inner = max(8, width // 2) if bottleneck is None else bottleneck
        self.temb = TimeEmbedding(hidden=tdim)
        self.inp = nn.Linear(dim, width)
        self.act = nn.SiLU()
        self.block_fc1 = nn.ModuleList([nn.Linear(width, inner) for _ in range(n_blocks)])
        self.block_fc2 = nn.ModuleList([nn.Linear(inner, width) for _ in range(n_blocks)])
        self.block_tproj = nn.ModuleList([nn.Linear(tdim, inner) for _ in range(n_blocks)])
        self.out = nn.Linear(width, dim)

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        temb = self.temb(t_norm)
        h = self.act(self.inp(x))
        for fc1, fc2, tproj in zip(self.block_fc1, self.block_fc2, self.block_tproj):
            skip = h
            h = fc1(h)
            h = self.act(h)
            h = h + tproj(temb)
            h = self.act(h)
            h = fc2(h)
            h = h + skip
        return self.out(h)
