"""Utility functions for diffusion model experiments."""

# Authors: Hamza Cherkaoui

import numpy as np
import torch


def set_seed(
    seed: int,
) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(
    force_cpu: bool,
) -> torch.device:
    """Select CUDA if available unless CPU is forced."""
    if torch.cuda.is_available() and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")


def format_tick(
    v: float,
    dec: int = 2,
    tol: float = 1e-6,
) -> str:
    """Format tick value: if integer-valued, format as int, else 3 significant digits."""
    if np.isfinite(v) and abs(v - round(v)) < tol:
        return str(int(round(v)))
    return f"{v:.{dec}g}"
