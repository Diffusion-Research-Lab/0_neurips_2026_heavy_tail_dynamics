"""Metrics module."""

# Authors: Hamza Cherkaoui

from typing import Callable, Union
import torch
from torch import Tensor, nn
from sampling import sample_from_flow_matching


def fid(x: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-6) -> torch.Tensor:
    """Compute Fréchet Inception Distance (FID) between two sets of samples.

    Paramters
    ---------
    x: Tensor
        Samples from distribution X, shape (N, D).
    y: Tensor
        Samples from distribution Y, shape (M, D).
    eps: float
        Small value to add to the diagonal of covariance matrices for numerical stability.

    Returns
    -------
    Tensor
        FID value as a scalar Tensor.
    """
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    x_centered = x - mu_x
    y_centered = y - mu_y

    n_x = x.shape[0]
    n_y = y.shape[0]
    cov_x = (x_centered.T @ x_centered) / (n_x - 1)
    cov_y = (y_centered.T @ y_centered) / (n_y - 1)

    cov_x = cov_x + eps * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    cov_y = cov_y + eps * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    cov_prod = cov_x @ cov_y
    cov_prod = 0.5 * (cov_prod + cov_prod.T)

    eigvals, eigvecs = torch.linalg.eigh(cov_prod)
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_cov_prod = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T

    return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(cov_x + cov_y - 2.0 * sqrt_cov_prod)


def evaluate(
    metric: Callable[[Tensor, Tensor], Tensor],
    base_data_or_base_sampler: Union[Tensor, Callable[[], Tensor]],
    ref_data_or_ref_sampler: Union[Tensor, Callable[[], Tensor]],
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Evaluate a flow model using a given metric.

    Parameters
    ----------
    metric:
        Function taking (x_real: Tensor, x_model: Tensor) and returning a scalar Tensor.
        Example: FID-like metric.
    data_or_sampler:
        Either:
          - Tensor of shape (N, D) with reference samples, or
          - Callable with no arguments returning a Tensor of shape (N, D).
    model:
        Flow model implementing v_theta(x, t) as model(x, t).
    device, dtype:
        Device and dtype for computation.
    n_steps, t0, t1:
        Euler integration parameters for sampling from the flow.

    Returns
    -------
    float
        Metric value as a Python float.
    """
    model.eval()

    with torch.no_grad():

        if isinstance(base_data_or_base_sampler, torch.Tensor):
            x_base = base_data_or_base_sampler.to(device=device, dtype=dtype)
        else:
            x_base = base_data_or_base_sampler()
            if not isinstance(x_base, torch.Tensor):
                raise TypeError("`ref_data_or_ref_sampler` callable must return a torch.Tensor.")
            x_base = x_base.to(device=device, dtype=dtype)

        if isinstance(ref_data_or_ref_sampler, torch.Tensor):
            x_ref = ref_data_or_ref_sampler.to(device=device, dtype=dtype)
        else:
            x_ref = ref_data_or_ref_sampler()
            if not isinstance(x_ref, torch.Tensor):
                raise TypeError("`ref_data_or_ref_sampler` callable must return a torch.Tensor.")
            x_ref = x_ref.to(device=device, dtype=dtype)

        if x_base.ndim != 2:
            raise ValueError(f"Expected base samples of shape (N, D), got {x_base.shape}.")

        if x_ref.ndim != 2:
            raise ValueError(f"Expected reference samples of shape (N, D), got {x_ref.shape}.")

        x_gen = sample_from_flow_matching(model=model, device=device, dtype=dtype, base_or_sample=x_base)

        m = metric(x_ref, x_gen)

        if not isinstance(m, torch.Tensor):
            return float(m)

        if m.numel() != 1:
            raise ValueError(f"Metric must return a scalar Tensor, got shape {m.shape}.")

        return float(m.detach().cpu().item())
