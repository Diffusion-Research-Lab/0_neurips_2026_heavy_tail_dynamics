"""Utility functions for diffusion model experiments."""

from typing import Any, MutableMapping
import math
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
def cosine_schedule(
    n_steps: int,
    device: torch.device,
    fdtype: torch.dtype,
    idtype: torch.dtype,
    s: float = 0.008,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return the cosine schedule.
    """
    t = torch.arange(0, n_steps + 1, device=device, dtype=idtype)

    alpha_bar = torch.cos(((t / n_steps) + s) / (1.0 + s) * (math.pi / 2.0)).pow(2).clamp(min=eps)
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha_bar = alpha_bar.to(device=device, dtype=fdtype)

    alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=eps)

    betas = 1 - alphas

    sqrt_post_var = torch.sqrt(betas * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:]))

    return alpha_bar[1:], alphas, betas, sqrt_post_var
