"""Loss module."""

# Authors: Hamza Cherkaoui

import torch


def lp_loss(y_true, y_pred, r=1.0):
    """Compute the L^p loss."""
    e = y_true - y_pred
    abs_e = e.abs()
    return abs_e.pow(r).mean().pow(1.0 / r)


def huber_loss(y_true, y_pred, delta=1.0):
    """Compute the Huber loss."""
    e = y_true - y_pred
    abs_e = e.abs()
    return torch.where(abs_e <= delta, 0.5 * e**2, delta * (abs_e - 0.5 * delta)).mean()
