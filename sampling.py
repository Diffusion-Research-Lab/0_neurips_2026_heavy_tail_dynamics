"""Sampling functions for various distributions."""

# Authors: Hamza Cherkaoui

import math
from typing import Union
import torch
import torch.nn as nn
from constants import EPS


def sample_alpha_stable(
    n_samples: int,
    dim: int,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Symmetric alpha-stable (β=0) sampler via Chambers-Mallows-Stuck."""
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (0, 2].")

    u = (torch.rand((n_samples, dim), device=device, dtype=dtype) - 0.5) * math.pi
    w = torch.empty((n_samples, dim), device=device, dtype=dtype).exponential_(1.0)

    if abs(alpha - 1.0) < 1e-12:
        return torch.tan(u)

    if abs(alpha - 2.0) < 1e-12:
        return math.sqrt(2.0) * torch.randn((n_samples, dim), device=device, dtype=dtype)

    cos_u = torch.cos(u).clamp_min(EPS)
    s1 = torch.sin(alpha * u) / (cos_u ** (1.0 / alpha))

    cos_term = torch.cos(u - alpha * u).clamp_min(EPS)
    s2 = (cos_term / w) ** ((1.0 - alpha) / alpha)

    return s1 * s2


def sample_student_t(
    n_samples: int,
    dim: int,
    nu: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Multivariate Student's t with identity scale and dof=nu."""
    z = torch.randn(n_samples, dim, device=device, dtype=dtype)
    s = torch.distributions.Gamma(nu / 2.0, 0.5).sample((n_samples,)).to(device=device, dtype=dtype)
    scale = torch.sqrt((s / nu).clamp_min(EPS)).unsqueeze(1)
    return z / scale


def sample_gaussian(
    n_samples: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard Gaussian N(0, I)."""
    return torch.randn(n_samples, dim, device=device, dtype=dtype)


def sample_bimodal_gaussian(
    n_samples: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bimodal Gaussian distribution."""
    centers = torch.tensor([-2.0, 2.0], device=device, dtype=dtype)
    idx = torch.randint(0, 2, (n_samples,), device=device)
    return centers[idx].unsqueeze(1) + 0.3 * torch.randn(n_samples, dim, device=device, dtype=dtype)


@torch.no_grad()
def sample_from_diffusion(
    model: nn.Module,
    n_samples: int,
    dim: int,
    n_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    beta_start: float = 1e-4,
    beta_end: float = 1e-2,
) -> torch.Tensor:
    """DDPM ancestral sampling for an x0-predicting model: x0_hat = model(x_t, t_norm)."""
    model.eval()

    betas = torch.linspace(beta_start, beta_end, n_steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    abar_prev = torch.cat([torch.ones(1, device=device, dtype=dtype), abar[:-1]])
    sqrt_abar = torch.sqrt(abar)
    sqrt_one_minus_abar = torch.sqrt(1.0 - abar)
    sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
    sqrt_post_var = torch.sqrt((betas * (1.0 - abar_prev) / (1.0 - abar).clamp_min(1e-12)).clamp_min(1e-20))

    x = torch.randn(n_samples, dim, device=device, dtype=dtype)

    for step in range(n_steps - 1, -1, -1):
        t_norm = torch.full((n_samples,), step / float(n_steps - 1), device=device, dtype=dtype)

        x0_hat = model(x, t_norm)
        eps_hat = (x - sqrt_abar[step] * x0_hat) / sqrt_one_minus_abar[step]

        x = sqrt_recip_alpha[step] * (x - (1.0 - alphas[step]) / sqrt_one_minus_abar[step] * eps_hat)
        if step > 0:
            x = x + sqrt_post_var[step] * torch.randn_like(x)

    return x


@torch.no_grad()
def sample_from_flow_matching(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    base_or_sample: Union[str, torch.Tensor] = "gaussian",
    n_samples: int = 1000,
    dim: int = 2,
    nu_source: float | None = None,
    steps: int = 300,
    t_min: float = 0.05,
    t_max: float = 0.95,
) -> torch.Tensor:
    """Deterministic FM sampling: integrate dx/dt = v_theta(x,t) from a base x0."""
    model.eval()

    if isinstance(base_or_sample, torch.Tensor):
        x = base_or_sample.to(device=device, dtype=dtype)
        if x.ndim != 2:
            raise ValueError(f"`base_or_sample` tensor must be 2D, got shape {x.shape}.")
    else:
        base = base_or_sample
        if base == "gaussian":
            x = sample_gaussian(n_samples=n_samples, dim=dim, device=device, dtype=dtype)
        elif base == "student_t":
            if nu_source is None:
                raise ValueError("nu_source must be provided when base='student_t'.")
            x = sample_student_t(nu=nu_source, n_samples=n_samples, dim=dim, device=device,
                                 dtype=dtype)
        else:
            raise ValueError("base_or_sample must be a Tensor, 'gaussian' or 'student_t'.")

    count, _ = x.shape

    dt = (t_max - t_min) / float(steps)
    t = torch.full((count, 1), t_min, device=device, dtype=dtype)

    for _ in range(steps):
        v = model(x, t)
        x = x + dt * v
        t = t + dt

    return x
