"""Sampling functions for various distributions."""

import numpy as np
import scipy
import torch


def sample_scaled_scalar_alpha_stable(
    n_samples: int,
    alpha: float,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample positive scalar mixing coefficients for isotropic alpha-stable sampling."""
    alpha = float(alpha)

    if not (0.0 < alpha <= 2.0):
        raise ValueError(f"`alpha` must be in (0,2], got {alpha}.")
    if alpha == 2.0:
        return (2.0 * torch.ones((n_samples, 1), device=device, dtype=dtype))

    scale = 2.0 * np.cos(np.pi * alpha / 4.0) ** (2.0 / alpha)
    draws = scipy.stats.levy_stable.rvs(alpha / 2.0, 1.0, loc=0.0, scale=scale, size=n_samples)

    return torch.as_tensor(draws, device=device, dtype=dtype).unsqueeze(-1)


def sample_scaled_isotropic_alpha_stable(
    n_samples: int,
    dim: int,
    alpha: float,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample isotropic alpha-stable vectors via Gaussian scale mixing.

    Draws
        G ~ N(0, I) in R^dim
    and
        A from sample_scaled_scalar_alpha_stable(...),
    then returns
        X = sqrt(A) * G.
    """
    G = torch.randn(n_samples, dim, device=device, dtype=dtype)
    A = sample_scaled_scalar_alpha_stable(n_samples, alpha=alpha, device=device, dtype=dtype)
    return A.sqrt() * G


def sample_spiral(
    n_samples: int,
    spiral_turns: float = 3.0,
    spiral_radius: float = 4.0,
    spiral_noise: float = 0.15,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a 2d spirale cloud points.
    """
    u = torch.rand((n_samples,), device=device, dtype=dtype)
    theta = (2.0 * np.pi * spiral_turns) * u
    r = spiral_radius * u

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    pts = torch.stack([x, y], dim=1)

    if spiral_noise > 0.0:
        pts += spiral_noise * torch.randn_like(pts)

    return pts


def sample_checker(
    n_samples: int,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a 2d checker cloud points.
    """
    x1 = torch.rand(n_samples, device=device, dtype=dtype) * 4 - 2
    x2_ = torch.rand(n_samples, device=device, dtype=dtype)
    x2_ -= torch.randint(high=2, size=(n_samples, ), device=device, dtype=dtype) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    pts = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return pts.float()


def sample_student_t(
    n_samples: int,
    dim: int,
    nu: float,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a multivariate Student's t with identity scale and dof=nu.
    """
    z = torch.randn(n_samples, dim, device=device, dtype=dtype)
    s = torch.distributions.Gamma(nu / 2.0, 0.5).sample((n_samples,)).to(device=device, dtype=dtype)
    scale = torch.sqrt((s / nu).clamp_min(torch.finfo(dtype).tiny)).unsqueeze(1)
    return z / scale


def sample_exponential(
    n_samples: int,
    dim: int,
    rate: float = 1.0,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample from an exponential distribution Exp(rate).
    """
    if rate <= 0:
        raise ValueError("rate must be > 0.")

    return torch.empty(n_samples, dim, device=device, dtype=dtype).exponential_(rate)


def sample_gaussian(
    n_samples: int,
    dim: int,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a standard Gaussian N(0, I).
    """
    return torch.randn(n_samples, dim, device=device, dtype=dtype)


def _sample_bimodal_gaussian(
    n_samples: int,
    dim: int,
    mu: float = 2.0,
    s: float = 0.5,
    p: float = 0.5,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Helper for sampling a bimodal Gaussian distribution.
    """
    centers = torch.tensor([-mu, mu], device=device, dtype=dtype)
    idx = torch.multinomial(torch.tensor([p, 1 - p]), n_samples, replacement=True)
    return centers[idx].unsqueeze(1) + s * torch.randn(n_samples, dim, device=device, dtype=dtype)


def sample_balanced_bimodal_gaussian(
    n_samples: int,
    dim: int,
    mu: float = 2.0,
    s: float = 0.6,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a balanced bimodal Gaussian distribution.
    """
    return _sample_bimodal_gaussian(n_samples=n_samples, dim=dim, mu=mu, s=s,
                                    p=0.5, device=device, dtype=dtype)


def sample_unbalanced_bimodal_gaussian(
    n_samples: int,
    dim: int,
    mu: float = 4.0,
    s: float = 0.3,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample an unbalanced bimodal Gaussian distribution.
    """
    return _sample_bimodal_gaussian(n_samples=n_samples, dim=dim, mu=mu, s=s,
                                    p=0.01, device=device, dtype=dtype)
