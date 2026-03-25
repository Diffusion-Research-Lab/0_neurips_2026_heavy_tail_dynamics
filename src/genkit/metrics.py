"""Classical evaluation metrics."""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

__all__ = [
    "gen_separability_roc_auc",
    "mmd_rbf",
    "mssle_90",
    "mssle_95",
    "sliced_wasserstein2",
    "wasserstein_distance",
]


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


def _reduce(x, reduction="mean"):
    """Reduce a tensor result to a scalar or leave it unchanged."""
    if reduction == "mean":
        return x.mean().item()
    if reduction == "sum":
        return x.sum().item()
    if reduction == "none":
        return x
    raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")


def _to_numpy_2d(x):
    """Convert array-like input to a detached 2D NumPy array."""
    x = _to_2d_tensor(x, dtype=torch.float64)
    return x.detach().cpu().numpy()


def _ssle_tail(x_ref, x_gen, xi=0.95, scale=1.0, n_grid=1000, eps=1e-12, reduction="mean"):
    """Compute squared signed log errors over upper-tail empirical quantiles."""
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


def mmd_rbf(x_ref, x_gen, gamma=None):
    """Compute the unbiased RBF-kernel MMD between two empirical samples."""
    x_ref = _to_numpy_2d(x_ref)
    x_gen = _to_numpy_2d(x_gen)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    n_ref, n_gen = x_ref.shape[0], x_gen.shape[0]
    if n_ref < 2 or n_gen < 2:
        raise ValueError("mmd_rbf requires at least two samples in each input.")

    if gamma is None:
        z = np.vstack([x_ref, x_gen])
        d2 = cdist(z, z, metric="sqeuclidean")
        positive_d2 = d2[d2 > 0.0]
        median_d2 = np.median(positive_d2) if positive_d2.size else 1.0
        gamma = 1.0 / max(2.0 * float(median_d2), np.finfo(float).eps)
    elif gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}.")

    k_xx = np.exp(-gamma * cdist(x_ref, x_ref, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(x_gen, x_gen, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(x_ref, x_gen, metric="sqeuclidean"))

    mmd2 = (k_xx.sum() - np.trace(k_xx)) / (n_ref * (n_ref - 1)) + (k_yy.sum() - np.trace(k_yy)) / (n_gen * (n_gen - 1)) - 2.0 * k_xy.mean()  # noqa
    return float(mmd2)


def gen_separability_roc_auc(x_ref, x_gen, train_size=0.5, seed=0, clf=None):
    """Estimate classifier separability between real and generated samples with ROC-AUC."""
    x_ref = _to_numpy_2d(x_ref)
    x_gen = _to_numpy_2d(x_gen)

    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    if not 0.0 < float(train_size) < 1.0:
        raise ValueError(f"train_size must be in (0, 1), got {train_size}.")
    if x_ref.shape[0] < 2 or x_gen.shape[0] < 2:
        raise ValueError("gen_separability_roc_auc requires at least two samples in each input.")

    xr_tr, xr_te = train_test_split(x_ref, train_size=train_size, random_state=seed, shuffle=True)
    xg_tr, xg_te = train_test_split(x_gen, train_size=train_size, random_state=seed, shuffle=True)
    x_train = np.vstack([xr_tr, xg_tr])
    y_train = np.r_[np.zeros(len(xr_tr), dtype=int), np.ones(len(xg_tr), dtype=int)]

    clf = clf or make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(x_train, y_train)

    if hasattr(clf, "predict_proba"):
        real_scores = clf.predict_proba(xr_te)[:, 1]
        gen_scores = clf.predict_proba(xg_te)[:, 1]
    else:
        real_scores = clf.predict(xr_te)
        gen_scores = clf.predict(xg_te)

    y_true = np.r_[np.zeros(len(real_scores), dtype=int), np.ones(len(gen_scores), dtype=int)]
    y_score = np.r_[real_scores, gen_scores]
    return float(roc_auc_score(y_true, y_score))


def _wasserstein_1d(x_ref, x_gen, p=1, n_grid=1000, eps=1e-12, reduction="mean"):
    """Compute 1D empirical p-Wasserstein distances from quantile functions."""
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
    """Approximate multivariate Wasserstein distances with Sinkhorn transport."""
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
