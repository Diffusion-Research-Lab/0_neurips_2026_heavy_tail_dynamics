"""Utility functions for diffusion model experiments."""

from typing import Any, MutableMapping
import math
import torch

_MISSING = object()


def getpop(
    d: MutableMapping[Any, Any],
    key: Any,
    default: Any = _MISSING,
) -> Any:
    """
    Return d[key] and delete the key if present; otherwise return default (or raise KeyError).
    """
    v = d.pop(key, _MISSING)
    if v is not _MISSING:
        return v
    if default is _MISSING:
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


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as '<h>h <min>min <s>s'."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}min")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


@torch.no_grad()
def cosine_schedule(
    n_steps: int,
    device: torch.device,
    fdtype: torch.dtype,
    idtype: torch.dtype,
    s: float = 0.008,
    eps: float = 1e-8,
    beta_max: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return the cosine schedule.
    """
    t = torch.arange(0, n_steps + 1, device=device, dtype=idtype)

    alpha_bar = torch.cos(((t / n_steps) + s) / (1.0 + s) * (math.pi / 2.0)).pow(2).clamp(min=eps)
    alpha_bar = (alpha_bar / alpha_bar[0]).to(device=device, dtype=fdtype)

    alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=eps)
    betas = (1.0 - alphas).clamp(min=eps, max=beta_max)
    alphas = 1.0 - betas

    alpha_bar = torch.cumprod(alphas, dim=0)  # to enforce the cumprod in the formula
    alpha_bar_prev = torch.cat([torch.ones(1, device=device, dtype=fdtype), alpha_bar[:-1]])
    sqrt_post_var = torch.sqrt(betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(eps))

    return alpha_bar, alphas, betas, sqrt_post_var
