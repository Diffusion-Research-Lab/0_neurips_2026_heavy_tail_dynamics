"""Utility functions for diffusion model experiments."""

# Authors: Hamza Cherkaoui

from typing import Any, MutableMapping
import math
import numpy as np
import torch


_sentinel = object()


def getpop(
    d: MutableMapping[Any, Any],
    key: Any,
    default: Any = _sentinel,
) -> Any:
    """
    Return d[key] and delete the key if present; otherwise return default (or raise KeyError).
    """
    v = d.pop(key, _sentinel)
    if v is not _sentinel:
        return v
    if default is _sentinel:
        raise KeyError(key)
    return default


def to_numpy(x):
    """Recursively convert torch tensors (incl. nested containers) to CPU NumPy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_numpy(v) for v in x]
    return x


@torch.no_grad()
def cosine_betas(
    n_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Return the beta schedule.
    """
    t = torch.arange(n_steps, device=device, dtype=dtype)
    alpha_bar = torch.cos(((t / n_steps) + s) / (1.0 + s) * (math.pi / 2.0)).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0].clamp_min(torch.finfo(dtype).eps)

    alpha_bar_prev = torch.cat([alpha_bar[:1], alpha_bar[:-1]])
    betas = 1.0 - (alpha_bar / alpha_bar_prev).clamp(0.0, 1.0)
    betas[0] = 0.0

    return betas.clamp(0.0)


@torch.no_grad()
def make_schedule(
    n_steps: int,
    device: torch.device | str,
    dtype: torch.dtype,
    s: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return the classic DDPM schedule.
    """
    betas = cosine_betas(n_steps, device=device, dtype=dtype, s=s)
    alphas = 1.0 - betas
    alpha_bar = torch.empty(n_steps + 1, device=device, dtype=dtype)
    alpha_bar[0] = 1.0
    alpha_bar[1:] = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bar
