"""Backward-compatible re-exports for inspect helpers."""

from .inspect import (
    err_time_grid,
    jacobian_spectral_at_time,
    jacobian_step_grid,
    knn_kl,
    mse_loss_at_batch,
    require_family,
    sample_forward_marginal,
    subsample_rows,
)

__all__ = [
    "require_family",
    "err_time_grid",
    "jacobian_step_grid",
    "jacobian_spectral_at_time",
    "subsample_rows",
    "sample_forward_marginal",
    "knn_kl",
    "mse_loss_at_batch",
]
