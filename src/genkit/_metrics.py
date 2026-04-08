"""Private metric helpers."""

import math
import numpy as np
from scipy import linalg
import torch


_EPS = 1e-12


def _to_tensor(x, device=None, dtype=torch.float64):
    """Convert array-like input to a tensor on the requested device and dtype."""
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return x.to(device=device, dtype=dtype)


def _to_2d_tensor(x, device=None, dtype=torch.float64):
    """Normalize array-like input to a non-empty 2D tensor."""
    x = _to_tensor(x, device=device, dtype=dtype)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Input must be non-empty.")
    return x


def _to_numpy_2d(x):
    """Convert array-like input to a detached 2D NumPy array."""
    x = _to_2d_tensor(x, dtype=torch.float64)
    return x.detach().cpu().numpy()


def _validate_same_feature_dim_numpy(x_ref, x_gen):
    """Validate that two arrays have the same feature dimension."""
    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")


def _reduce(x, reduction="mean"):
    """Reduce a tensor result to a scalar or leave it unchanged."""
    if reduction == "mean":
        return x.mean().item()
    if reduction == "sum":
        return x.sum().item()
    if reduction == "none":
        return x
    raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")


def _validate_same_feature_dim(x_ref, x_gen):
    """Validate that two tensors have the same feature dimension."""
    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")


def _validate_quantile_grid(n_grid, eps):
    """Validate quantile-grid parameters."""
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must satisfy 0 < eps < 1.")


def _sslog1p(x, scale=1.0):
    """Signed log1p transform used by SSLE-type metrics."""
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive.")
    return torch.sign(x) * torch.log1p(torch.abs(x) / scale)


def _quantile_grid(n_grid=1000, eps=1e-12, *, device=None, dtype=None):
    """Return a uniform quantile grid on [0, 1 - eps]."""
    _validate_quantile_grid(n_grid=n_grid, eps=eps)
    return torch.linspace(0.0, 1.0 - eps, n_grid, device=device, dtype=dtype)


def _upper_tail_grid(xi=0.95, n_grid=1000, eps=1e-12, *, device=None, dtype=None):
    """Return a uniform upper-tail quantile grid on [xi, 1 - eps]."""
    _validate_quantile_grid(n_grid=n_grid, eps=eps)
    if not 0.0 <= xi < 1.0:
        raise ValueError("xi must satisfy 0 <= xi < 1.")
    if xi >= 1.0 - eps:
        raise ValueError("xi must be strictly smaller than 1 - eps.")
    return torch.linspace(xi, 1.0 - eps, n_grid, device=device, dtype=dtype)


def _upper_tail_log_grid(
    n_samples,
    u_min=None,
    u_max=1e-1,
    n_grid=128,
    k_min=10,
    *,
    device=None,
    dtype=None,
):
    """Return a log-spaced grid of upper-tail probabilities u = 1 - p.

    The lower endpoint is clipped so that the extreme empirical quantiles are not
    deeper than roughly ``k_min`` exceedances for ``n_samples`` observations.
    """
    if n_samples < 2:
        raise ValueError("Need at least two samples to build a tail grid.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    if k_min <= 0:
        raise ValueError("k_min must be strictly positive.")
    if not 0.0 < u_max < 1.0:
        raise ValueError("u_max must satisfy 0 < u_max < 1.")

    empirical_u_min = max(float(k_min) / float(n_samples), _EPS)
    if u_min is None:
        u_min = empirical_u_min
    else:
        u_min = max(float(u_min), empirical_u_min)

    if not 0.0 < u_min < u_max < 1.0:
        raise ValueError("Need 0 < u_min < u_max < 1.")

    log_u = torch.linspace(
        math.log10(u_min),
        math.log10(u_max),
        n_grid,
        device=device,
        dtype=dtype,
    )
    base = torch.tensor(10.0, device=device, dtype=dtype)
    u = torch.pow(base, log_u)
    p = 1.0 - u
    return p, u, log_u


def _quantile_values(x, p):
    """Evaluate empirical quantiles of each feature at probabilities p."""
    x = _to_2d_tensor(x)
    return torch.quantile(x, p, dim=0)


def _upper_tail_quantiles(x_ref, x_gen, xi=0.95, n_grid=1000, eps=1e-12, return_p=False):
    """Return upper-tail empirical quantile grids for two samples."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    p = _upper_tail_grid(xi=xi, n_grid=n_grid, eps=eps, device=x_ref.device, dtype=x_ref.dtype)
    q_ref = torch.quantile(x_ref, p, dim=0)
    q_gen = torch.quantile(x_gen, p, dim=0)
    if return_p:
        return q_ref, q_gen, p
    return q_ref, q_gen


def _upper_tail_quantiles_logspaced(
    x_ref,
    x_gen,
    u_min=None,
    u_max=1e-1,
    n_grid=128,
    k_min=10,
    return_grid=False,
):
    """Return upper-tail empirical quantiles on a log-spaced tail grid."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    n_samples = min(x_ref.shape[0], x_gen.shape[0])
    p, u, log_u = _upper_tail_log_grid(
        n_samples=n_samples,
        u_min=u_min,
        u_max=u_max,
        n_grid=n_grid,
        k_min=k_min,
        device=x_ref.device,
        dtype=x_ref.dtype,
    )
    q_ref = torch.quantile(x_ref, p, dim=0)
    q_gen = torch.quantile(x_gen, p, dim=0)
    if return_grid:
        return q_ref, q_gen, p, u, log_u
    return q_ref, q_gen


def _wasserstein_1d(x_ref, x_gen, p=1, n_grid=1000, eps=1e-12, reduction="mean"):
    """Compute 1D empirical p-Wasserstein distances from quantile functions."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if p < 1:
        raise ValueError("p must satisfy p >= 1.")

    q = _quantile_grid(n_grid=n_grid, eps=eps, device=x_ref.device, dtype=x_ref.dtype)
    q_ref = torch.quantile(x_ref, q, dim=0)
    q_gen = torch.quantile(x_gen, q, dim=0)
    scores = torch.trapz((q_ref - q_gen).abs().pow(p), q, dim=0).pow(1.0 / p)
    return _reduce(scores, reduction=reduction)


def _sinkhorn_wasserstein(x_ref, x_gen, p=1, reg=1e-2, n_iters=200, tol=1e-7):
    """Approximate multivariate Wasserstein distances with Sinkhorn transport."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if p < 1:
        raise ValueError("p must satisfy p >= 1.")
    if reg <= 0.0:
        raise ValueError("reg must be strictly positive.")
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1.")
    if tol <= 0.0:
        raise ValueError("tol must be strictly positive.")

    n_ref, n_gen = x_ref.shape[0], x_gen.shape[0]
    a = torch.full((n_ref,), 1.0 / n_ref, device=x_ref.device, dtype=x_ref.dtype)
    b = torch.full((n_gen,), 1.0 / n_gen, device=x_ref.device, dtype=x_ref.dtype)
    cost = torch.cdist(x_ref, x_gen, p=2).pow(p)
    K = torch.exp(-cost / reg).clamp_min(tol)
    u, v = torch.ones_like(a), torch.ones_like(b)

    for _ in range(n_iters):
        u_prev = u
        u = a / (K @ v).clamp_min(tol)
        v = b / (K.t() @ u).clamp_min(tol)
        if (u - u_prev).abs().max() < tol:
            break

    pi = u[:, None] * K * v[None, :]
    return (pi.mul(cost).sum().clamp_min(0.0)).pow(1.0 / p).item()


def _sqeuclidean_cdist(x_ref, x_gen):
    """Compute squared Euclidean cross-distances between two NumPy arrays."""
    x_ref_norm = np.sum(x_ref * x_ref, axis=1, keepdims=True)
    x_gen_norm = np.sum(x_gen * x_gen, axis=1, keepdims=True).T
    d2 = x_ref_norm + x_gen_norm - 2.0 * x_ref @ x_gen.T
    return np.maximum(d2, 0.0)


def _median_heuristic_gamma(x_ref, x_gen):
    """Return an RBF bandwidth from the median pairwise squared distance."""
    stacked = np.vstack([x_ref, x_gen])
    d2 = _sqeuclidean_cdist(stacked, stacked)
    positive_d2 = d2[d2 > 0.0]
    median_d2 = np.median(positive_d2) if positive_d2.size else 1.0
    return 1.0 / max(2.0 * float(median_d2), np.finfo(float).eps)


def _rbf_kernel_matrix(x_ref, x_gen, gamma):
    """Compute the RBF Gram matrix between two NumPy arrays."""
    return np.exp(-gamma * _sqeuclidean_cdist(x_ref, x_gen))


def _mmd2_from_gram_matrices(k_xx, k_yy, k_xy, *, estimator):
    """Compute an empirical squared MMD from precomputed Gram matrices."""
    n_ref = k_xx.shape[0]
    n_gen = k_yy.shape[0]

    if estimator == "biased":
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    if estimator == "unbiased":
        return (k_xx.sum() - np.trace(k_xx)) / (n_ref * (n_ref - 1)) + (k_yy.sum() - np.trace(k_yy)) / (n_gen * (n_gen - 1)) - 2.0 * k_xy.mean()
    raise ValueError("estimator must be one of {'biased', 'unbiased'}.")


def _feature_mean_and_covariance(x):
    """Return the empirical mean vector and covariance matrix of one 2D array."""
    mean = np.mean(x, axis=0)
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    return mean, cov


def _frechet_gaussian_distance(mean_ref, cov_ref, mean_gen, cov_gen, eps=1e-6):
    """Compute the Fréchet distance between two fitted Gaussian laws."""
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive.")

    mean_diff = mean_ref - mean_gen
    cov_prod_sqrt = linalg.sqrtm(cov_ref @ cov_gen)
    if not np.isfinite(cov_prod_sqrt).all():
        eye = np.eye(cov_ref.shape[0], dtype=cov_ref.dtype)
        cov_prod_sqrt = linalg.sqrtm((cov_ref + eps * eye) @ (cov_gen + eps * eye))
    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real

    return float(mean_diff @ mean_diff + np.trace(cov_ref + cov_gen - 2.0 * cov_prod_sqrt))
