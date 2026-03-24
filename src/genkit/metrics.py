"""Classical evaluation metrics."""

import torch


__all__ = [
    "mssle_90",
    "mssle_95",
    "wasserstein_distance",
    "sliced_wasserstein2",
]


def _to_tensor(x, device=None, dtype=torch.float64):
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return x.to(device=device, dtype=dtype)


def _to_2d_tensor(x, device=None, dtype=torch.float64):
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


def _reduce(x, reduction="mean"):
    if reduction == "mean":
        return x.mean().item()
    if reduction == "sum":
        return x.sum().item()
    if reduction == "none":
        return x
    raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")


def _ssle_tail(x_ref, x_gen, xi=0.95, scale=1.0, n_grid=1000, eps=1e-12, reduction="mean"):
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    if not 0.0 <= xi < 1.0:
        raise ValueError("xi must satisfy 0 <= xi < 1.")
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive.")

    p = torch.linspace(xi, 1.0 - eps, n_grid, device=x_ref.device, dtype=x_ref.dtype)
    q_ref, q_gen = torch.quantile(x_ref, p, dim=0), torch.quantile(x_gen, p, dim=0)
    g_ref = torch.sign(q_ref) * torch.log1p(torch.abs(q_ref) / scale)
    g_gen = torch.sign(q_gen) * torch.log1p(torch.abs(q_gen) / scale)
    scores = torch.trapz((g_ref - g_gen).pow(2), p, dim=0)

    return _reduce(scores, reduction=reduction)


def mssle_90(x_ref, x_gen, n_grid=1000, eps=1e-12, reduction="mean"):
    """Compute the tail mean squared log error between two empirical distributions above the 0.90 quantile."""
    return _ssle_tail(x_ref=x_ref, x_gen=x_gen, xi=0.90, n_grid=n_grid, eps=eps, reduction=reduction)


def mssle_95(x_ref, x_gen, n_grid=1000, eps=1e-12, reduction="mean"):
    """Compute the tail mean squared log error between two empirical distributions above the 0.95 quantile."""
    return _ssle_tail(x_ref=x_ref, x_gen=x_gen, xi=0.95, n_grid=n_grid, eps=eps, reduction=reduction)


def _wasserstein_1d(x_ref, x_gen, p=1, n_grid=1000, eps=1e-12, reduction="mean"):
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    if p < 1:
        raise ValueError("p must satisfy p >= 1.")

    q = torch.linspace(0.0, 1.0 - eps, n_grid, device=x_ref.device, dtype=x_ref.dtype)
    q_ref, q_gen = torch.quantile(x_ref, q, dim=0), torch.quantile(x_gen, q, dim=0)
    scores = torch.trapz((q_ref - q_gen).abs().pow(p), q, dim=0).pow(1.0 / p)

    return _reduce(scores, reduction=reduction)


def _sinkhorn_wasserstein(x_ref, x_gen, p=1, reg=1e-2, n_iters=200, tol=1e-7):
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    if p < 1:
        raise ValueError("p must satisfy p >= 1.")

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


def wasserstein_distance(x_ref, x_gen, p=1, n_grid=1000, eps=1e-12, reg=1e-2, n_iters=200, tol=1e-7, reduction="mean"):
    """Compute the p-Wasserstein distance between two empirical distributions in 1D or higher dimensions."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    if x_ref.shape[1] == 1:
        return _wasserstein_1d(x_ref=x_ref, x_gen=x_gen, p=p, n_grid=n_grid, eps=eps, reduction=reduction)

    return _sinkhorn_wasserstein(x_ref=x_ref, x_gen=x_gen, p=p, reg=reg, n_iters=n_iters, tol=tol)


def sliced_wasserstein2(x_ref, x_gen, n_projections=128, n_grid=1000, eps=1e-12, seed=None):
    """Compute the sliced 2-Wasserstein distance by averaging 1D projected Wasserstein-2 costs."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")

    d = x_ref.shape[1]

    generator = None if seed is None else torch.Generator(device=x_ref.device).manual_seed(seed)
    dirs = torch.randn(n_projections, d, device=x_ref.device, dtype=x_ref.dtype, generator=generator)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=1, keepdim=True).clamp_min(1e-12)
    proj_ref, proj_gen = x_ref @ dirs.t(), x_gen @ dirs.t()

    q = torch.linspace(0.0, 1.0 - eps, n_grid, device=x_ref.device, dtype=x_ref.dtype)
    q_ref, q_gen = torch.quantile(proj_ref, q, dim=0), torch.quantile(proj_gen, q, dim=0)

    return torch.trapz((q_ref - q_gen).pow(2), q, dim=0).mean().item()
