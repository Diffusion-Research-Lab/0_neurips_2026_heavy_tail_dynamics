"""Utility functions for diffusion model experiments."""

# Authors: Hamza Cherkaoui

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
