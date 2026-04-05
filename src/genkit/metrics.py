"""Public metric functions."""

import numpy as np
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ._metrics import (
    _mmd2_from_gram_matrices,
    _median_heuristic_gamma,
    _rbf_kernel_matrix,
    _reduce,
    _sinkhorn_wasserstein,
    _sslog1p,
    _to_2d_tensor,
    _to_numpy_2d,
    _upper_tail_quantiles,
    _upper_tail_quantiles_logspaced,
    _validate_same_feature_dim,
    _validate_same_feature_dim_numpy,
    _wasserstein_1d,
)

__all__ = [
    "gen_separability_roc_auc",
    "metric_on_quantile",
    "mmd_rbf",
    "mssle",
    "mssle_paired",
    "sinkhorn_distance",
    "sliced_wasserstein2",
    "ssle_tail",
    "ssle_tail_curve",
    "wasserstein_distance",
]


def mssle_paired(x_ref, x_gen, scale=1.0, reduction="mean"):
    """Compute paired mean signed squared log error.

    This is a row-wise loss, not a permutation-invariant distributional metric.
    It is only meaningful when rows of ``x_ref`` and ``x_gen`` are aligned.
    """
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    if x_ref.shape != x_gen.shape:
        raise ValueError("x_ref and x_gen must have the same shape for mssle_paired.")

    g_ref = _sslog1p(x_ref, scale=scale)
    g_gen = _sslog1p(x_gen, scale=scale)
    scores = (g_ref - g_gen).pow(2).mean(dim=0)
    return _reduce(scores, reduction=reduction)


def mssle(x_ref, x_gen, scale=1.0, reduction="mean"):
    """Backward-compatible alias for :func:`mssle_paired`."""
    return mssle_paired(x_ref=x_ref, x_gen=x_gen, scale=scale, reduction=reduction)


def metric_on_quantile(metric_fn, x_ref, x_gen, xi=0.95, n_grid=1000, eps=1e-12, **metric_kwargs):
    """Apply a deterministic curve metric to upper-tail empirical quantiles.

    This wrapper is intended for metrics that compare aligned function values,
    such as Lp-type discrepancies. It is not recommended for sample-based
    metrics such as MMD or classifier two-sample tests.
    """
    q_ref, q_gen = _upper_tail_quantiles(x_ref, x_gen, xi=xi, n_grid=n_grid, eps=eps)
    return metric_fn(q_ref, q_gen, **metric_kwargs)


def ssle_tail(x_ref, x_gen, xi=0.95, scale=1.0, n_grid=1000, eps=1e-12, reduction="mean"):
    """Compute the interval-normalized upper-tail SSLE.

    This returns the average squared signed log-quantile error on the interval
    ``p in [xi, 1 - eps]``:

        (1 / ((1 - eps) - xi)) * ∫ (g(Q_ref(p)) - g(Q_gen(p)))^2 dp,

    where ``g(x) = sign(x) * log(1 + |x| / scale)``.
    """
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    q_ref, q_gen, p = _upper_tail_quantiles(x_ref, x_gen, xi=xi, n_grid=n_grid, eps=eps, return_p=True)
    err = (_sslog1p(q_ref, scale=scale) - _sslog1p(q_gen, scale=scale)).pow(2)
    interval = (p[-1] - p[0]).clamp_min(torch.finfo(p.dtype).eps)
    scores = torch.trapz(err, p, dim=0) / interval
    return _reduce(scores, reduction=reduction)


def ssle_tail_curve(
    x_ref,
    x_gen,
    u_min=None,
    u_max=1e-1,
    n_grid=128,
    scale=1.0,
    k_min=10,
    reduction="mean",
):
    """Return a pointwise upper-tail SSLE curve on a log-spaced tail grid.

    The grid is built in terms of ``u = 1 - p`` and clipped so that the smallest
    empirical tail probability is not deeper than roughly ``k_min`` exceedances.

    Returns
    -------
    u : torch.Tensor of shape [n_grid]
        Tail probabilities, ordered from smaller to larger values.
    err : float or torch.Tensor
        Pointwise squared signed log-quantile errors. If ``reduction='none'``,
        the shape is ``[n_grid, d]``. Otherwise the feature dimension is reduced.
    """
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    q_ref, q_gen, _p, u, _log_u = _upper_tail_quantiles_logspaced(
        x_ref,
        x_gen,
        u_min=u_min,
        u_max=u_max,
        n_grid=n_grid,
        k_min=k_min,
        return_grid=True,
    )
    err = (_sslog1p(q_ref, scale=scale) - _sslog1p(q_gen, scale=scale)).pow(2)
    return u, _reduce(err, reduction=reduction)


def mmd_rbf(x_ref, x_gen, gamma=None, estimator="biased"):
    """Compute the empirical squared RBF-kernel MMD between two samples.

    Parameters
    ----------
    x_ref, x_gen :
        Input samples with the same feature dimension.
    gamma : float or None
        RBF kernel parameter. If ``None``, use the median heuristic on the
        pooled pairwise squared distances.
    estimator : {"biased", "unbiased"}
        Empirical MMD estimator. The default ``"biased"`` estimator is always
        non-negative. The ``"unbiased"`` estimator can be slightly negative in
        finite samples.
    """
    x_ref = _to_numpy_2d(x_ref)
    x_gen = _to_numpy_2d(x_gen)
    _validate_same_feature_dim_numpy(x_ref, x_gen)

    n_ref, n_gen = x_ref.shape[0], x_gen.shape[0]
    if n_ref < 2 or n_gen < 2:
        raise ValueError("mmd_rbf requires at least two samples in each input.")

    if gamma is None:
        gamma = _median_heuristic_gamma(x_ref, x_gen)
    elif gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}.")

    k_xx = _rbf_kernel_matrix(x_ref, x_ref, gamma)
    k_yy = _rbf_kernel_matrix(x_gen, x_gen, gamma)
    k_xy = _rbf_kernel_matrix(x_ref, x_gen, gamma)
    mmd2 = _mmd2_from_gram_matrices(k_xx, k_yy, k_xy, estimator=estimator)
    return float(mmd2)


def gen_separability_roc_auc(x_ref, x_gen, train_size=0.5, seed=0, clf=None, n_repeats=1):
    """Estimate classifier separability between real and generated samples.

    When ``n_repeats > 1``, the function averages the ROC-AUC over repeated
    random train/test splits.
    """
    x_ref = _to_numpy_2d(x_ref)
    x_gen = _to_numpy_2d(x_gen)
    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")
    if not 0.0 < float(train_size) < 1.0:
        raise ValueError(f"train_size must be in (0, 1), got {train_size}.")
    if x_ref.shape[0] < 2 or x_gen.shape[0] < 2:
        raise ValueError("gen_separability_roc_auc requires at least two samples in each input.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    base_clf = clf or make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    aucs = []

    for repeat in range(n_repeats):
        cur_seed = None if seed is None else int(seed) + repeat
        xr_tr, xr_te = train_test_split(
            x_ref, train_size=train_size, random_state=cur_seed, shuffle=True
        )
        xg_tr, xg_te = train_test_split(
            x_gen, train_size=train_size, random_state=cur_seed, shuffle=True
        )

        x_train = np.vstack([xr_tr, xg_tr])
        y_train = np.r_[np.zeros(len(xr_tr), dtype=int), np.ones(len(xg_tr), dtype=int)]

        cur_clf = (
            clone(base_clf)
            if clf is not None
            else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        )
        cur_clf.fit(x_train, y_train)

        if hasattr(cur_clf, "predict_proba"):
            real_scores = cur_clf.predict_proba(xr_te)[:, 1]
            gen_scores = cur_clf.predict_proba(xg_te)[:, 1]
        else:
            real_scores = (
                cur_clf.decision_function(xr_te)
                if hasattr(cur_clf, "decision_function")
                else cur_clf.predict(xr_te)
            )
            gen_scores = (
                cur_clf.decision_function(xg_te)
                if hasattr(cur_clf, "decision_function")
                else cur_clf.predict(xg_te)
            )

        y_true = np.r_[np.zeros(len(real_scores), dtype=int), np.ones(len(gen_scores), dtype=int)]
        y_score = np.r_[real_scores, gen_scores]
        aucs.append(roc_auc_score(y_true, y_score))

    return float(np.mean(aucs))


def sinkhorn_distance(x_ref, x_gen, p=1, reg=1e-2, n_iters=200, tol=1e-7):
    """Compute the entropically regularized p-Wasserstein distance."""
    return _sinkhorn_wasserstein(x_ref=x_ref, x_gen=x_gen, p=p, reg=reg, n_iters=n_iters, tol=tol)


def wasserstein_distance(
    x_ref,
    x_gen,
    p=1,
    n_grid=1000,
    eps=1e-12,
    reg=1e-2,
    n_iters=200,
    tol=1e-7,
    reduction="mean",
):
    """Compute a p-Wasserstein-type distance between two empirical distributions.

    In one dimension, this is the exact empirical Wasserstein distance computed
    from quantile functions. In dimension ``d > 1``, this returns the entropically
    regularized Sinkhorn approximation.
    """
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    if x_ref.shape[1] == 1:
        return _wasserstein_1d(
            x_ref=x_ref, x_gen=x_gen, p=p, n_grid=n_grid, eps=eps, reduction=reduction
        )
    return _sinkhorn_wasserstein(x_ref=x_ref, x_gen=x_gen, p=p, reg=reg, n_iters=n_iters, tol=tol)


def sliced_wasserstein2(x_ref, x_gen, n_projections=128, n_grid=1000, eps=1e-12, seed=None):
    """Compute the sliced squared 2-Wasserstein distance.

    The return value is the average projected ``W2^2``, not its square root.
    """
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    if n_projections < 1:
        raise ValueError("n_projections must be at least 1.")

    d = x_ref.shape[1]
    generator = None if seed is None else torch.Generator(device=x_ref.device).manual_seed(seed)
    dirs = torch.randn(n_projections, d, device=x_ref.device, dtype=x_ref.dtype, generator=generator)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=1, keepdim=True).clamp_min(
        torch.finfo(x_ref.dtype).eps
    )

    proj_ref = x_ref @ dirs.t()
    proj_gen = x_gen @ dirs.t()
    q = torch.linspace(0.0, 1.0 - eps, n_grid, device=x_ref.device, dtype=x_ref.dtype)
    q_ref = torch.quantile(proj_ref, q, dim=0)
    q_gen = torch.quantile(proj_gen, q, dim=0)
    return torch.trapz((q_ref - q_gen).pow(2), q, dim=0).mean().item()
