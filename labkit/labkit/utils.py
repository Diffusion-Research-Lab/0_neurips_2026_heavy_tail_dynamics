"""Utility functions for diffusion model experiments."""

# Authors: Hamza Cherkaoui

import random
import numpy as np
import torch


def set_seed(
    seed: int,
    deterministic: bool = True,
) -> None:

    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(
    device: torch.device,
) -> torch.device:
    """
    Select CUDA if available unless CPU is forced.
    """
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)
