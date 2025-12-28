"""Sampling functions for various distributions."""

# Authors: Hamza Cherkaoui

import math
import torch


def sample_scalar_alpha_stable(
    n_samples: int,
    alpha: float,
    scale: float,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample an alpha-stable random scalar:
        A ~ S_{alpha,1}(0, scale)
    """
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0,1), got {alpha}.")
    if float(scale) <= 0.0:
        raise ValueError(f"`scale` must be > 0, got {scale}.")

    tiny = torch.finfo(dtype).tiny
    U = torch.empty((n_samples,), device=device, dtype=dtype).uniform_(tiny, math.pi - tiny)
    W = torch.empty((n_samples,), device=device, dtype=dtype).exponential_().clamp_min(tiny)

    sinU = torch.sin(U).clamp_min(tiny)
    SaU = torch.sin(alpha * U).clamp_min(tiny)
    S1aU = torch.sin((1.0 - alpha) * U).clamp_min(tiny)

    A0 = (SaU / (sinU ** (1.0 / alpha))) * ((S1aU / W) ** ((1.0 - alpha) / alpha))

    cos_fac = math.cos(math.pi * alpha / 2.0)
    if cos_fac <= 0.0:
        raise ValueError("cos(pi*alpha/2) must be positive for alpha in (0,1).")

    scale_t = torch.as_tensor(scale, device=device, dtype=dtype)
    mult = scale_t / (cos_fac ** (1.0 / alpha))
    return A0 * mult


def sample_scaled_scalar_alpha_stable(
    n_samples: int,
    alpha: float,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample an scaled alpha-stable random scalar:
        A ~ S_{alpha/2,1}(0, cos^{2/alpha}(pi alpha / 4))
    """
    alpha = float(alpha)
    c_A = 2.0 * (math.cos(math.pi * alpha / 4.0) ** (2.0 / alpha))
    A = sample_scalar_alpha_stable(n_samples=n_samples, alpha=alpha / 2,
                                   scale=c_A, device=device, dtype=dtype)

    return A.clamp_min(torch.finfo(dtype).eps).unsqueeze(-1)


def sample_scaled_isotropic_alpha_stable(
    n_samples: int,
    dim: int,
    alpha: float,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample an scaled alpha-stable random scalar: X = sqrt(A) G
        A ~ S_{alpha/2,1}(0, cos^2(pi alpha / 4)) and G ~ N(O, I)
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
    theta = (2.0 * math.pi * spiral_turns) * u
    r = spiral_radius * u

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    pts = torch.stack([x, y], dim=1)

    if spiral_noise > 0.0:
        pts += spiral_noise * torch.randn_like(pts)

    return pts


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
