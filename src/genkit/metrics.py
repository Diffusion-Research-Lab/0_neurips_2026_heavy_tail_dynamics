"""Metric functions module."""

import torch

__all__ = [
    "fid",
    "mmd_rbf",
    "mssle",
    "sliced_wasserstein",
    "tail_coverage_curve",
    "tail_coverage_error",
]


def _to_tensor(x, *, device=None, dtype=torch.float64):
    """Convert array-like input to a tensor on the requested device and dtype."""
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return x.to(device=device, dtype=dtype)


def _to_2d_tensor(x, *, device=None, dtype=torch.float64):
    """Normalize array-like input to a non-empty 2D tensor."""
    x = _to_tensor(x, device=device, dtype=dtype)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.numel() == 0 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("input must be non-empty")
    return x


def _validate_same_feature_dim(x_ref, x_gen):
    """Ensure both sample sets share the same feature dimension."""
    if x_ref.shape[1] != x_gen.shape[1]:
        raise ValueError("x_ref and x_gen must have the same feature dimension.")


def _rbf_kernel_matrix(x, y, gamma):
    """Evaluate an RBF kernel matrix."""
    return torch.exp(-float(gamma) * torch.cdist(x, y, p=2).pow(2))


def _median_heuristic_gamma(x_ref, x_gen):
    """Median-heuristic bandwidth for an RBF kernel."""
    z = torch.cat([x_ref, x_gen], dim=0)
    d2 = torch.cdist(z, z, p=2).pow(2)
    positive = d2[d2 > 0]
    if positive.numel() == 0:
        return 1.0
    median_d2 = torch.median(positive)
    return float(0.5 / median_d2.clamp_min(torch.finfo(z.dtype).eps).item())


def _feature_mean_and_covariance(x):
    """Fit an empirical Gaussian model to feature samples."""
    x = _to_2d_tensor(x)
    mean = x.mean(dim=0)
    centered = x - mean
    denom = max(int(x.shape[0]) - 1, 1)
    cov = centered.T @ centered / float(denom)
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def _matrix_sqrt_psd(mat):
    """Stable symmetric square root for a PSD matrix."""
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(0.0)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.T


def _frechet_gaussian_distance(mean_ref, cov_ref, mean_gen, cov_gen, eps=1e-6):
    """Compute the Fréchet distance between two fitted Gaussian laws."""
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive.")

    mean_ref = _to_2d_tensor(mean_ref, dtype=torch.float64).reshape(-1)
    mean_gen = _to_2d_tensor(mean_gen, device=mean_ref.device, dtype=mean_ref.dtype).reshape(-1)
    cov_ref = _to_2d_tensor(cov_ref, device=mean_ref.device, dtype=mean_ref.dtype)
    cov_gen = _to_2d_tensor(cov_gen, device=mean_ref.device, dtype=mean_ref.dtype)
    if cov_ref.shape != cov_gen.shape:
        raise ValueError("cov_ref and cov_gen must have the same shape.")

    eye = torch.eye(cov_ref.shape[0], device=cov_ref.device, dtype=cov_ref.dtype)
    cov_ref = 0.5 * (cov_ref + cov_ref.T) + float(eps) * eye
    cov_gen = 0.5 * (cov_gen + cov_gen.T) + float(eps) * eye

    cov_ref_sqrt = _matrix_sqrt_psd(cov_ref)
    middle = cov_ref_sqrt @ cov_gen @ cov_ref_sqrt
    middle_sqrt = _matrix_sqrt_psd(middle)
    mean_diff = mean_ref - mean_gen
    score = mean_diff.dot(mean_diff) + torch.trace(cov_ref + cov_gen - 2.0 * middle_sqrt)
    return float(score.clamp_min(0.0).item())


def _default_tail_probs(n, *, device, dtype, min_exceedances=10):
    """Choose a small default tail-probability grid with enough exceedances."""
    if n < 2:
        raise ValueError("need at least two samples")
    if min_exceedances <= 0:
        raise ValueError("min_exceedances must be strictly positive")
    p_min = max(float(min_exceedances) / float(n), 1e-12)
    base = torch.tensor([1e-1, 5e-2, 1e-2, 5e-3, 1e-3], device=device, dtype=dtype)
    probs = base[base >= p_min]
    if probs.numel() == 0:
        probs = torch.tensor([p_min], device=device, dtype=dtype)
    return probs


def fid(x_ref, x_gen, eps=1e-6):
    """Compute a Gaussian Fréchet distance between two feature samples."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if x_ref.shape[0] < 2 or x_gen.shape[0] < 2:
        raise ValueError("fid requires at least two samples in each input.")

    mean_ref, cov_ref = _feature_mean_and_covariance(x_ref)
    mean_gen, cov_gen = _feature_mean_and_covariance(x_gen)
    return _frechet_gaussian_distance(mean_ref, cov_ref, mean_gen, cov_gen, eps=eps)


def mssle(x_ref, x_gen, scale=1.0, reduction="mean"):
    """Compute SSLE on rank-aligned samples or quantile curves."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    n_points = min(int(x_ref.shape[0]), int(x_gen.shape[0]))
    if n_points < 2:
        raise ValueError("aligned order statistics require at least two samples in each input.")
    if x_ref.shape[0] == x_gen.shape[0]:
        x_ref = torch.sort(x_ref, dim=0).values
        x_gen = torch.sort(x_gen, dim=0).values
    else:
        p = torch.linspace(0.0, 1.0, n_points, device=x_ref.device, dtype=x_ref.dtype)
        x_ref = torch.quantile(x_ref, p, dim=0)
        x_gen = torch.quantile(x_gen, p, dim=0)

    g_ref = torch.sign(x_ref) * torch.log1p(torch.abs(x_ref) / float(scale))
    g_gen = torch.sign(x_gen) * torch.log1p(torch.abs(x_gen) / float(scale))
    errors = (g_ref - g_gen).pow(2).mean(dim=0)

    if reduction == "mean":
        return float(errors.mean().item())
    if reduction == "sum":
        return float(errors.sum().item())
    if reduction == "none":
        return errors
    raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")


def mmd_rbf(x_ref, x_gen, gamma=None, estimator="biased"):
    """Compute the empirical squared RBF-kernel MMD between two samples."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if min(int(x_ref.shape[0]), int(x_gen.shape[0])) < 2:
        raise ValueError("mmd_rbf requires at least two samples in each input.")

    if gamma is None:
        gamma = _median_heuristic_gamma(x_ref, x_gen)
    elif float(gamma) <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}.")

    k_xx = _rbf_kernel_matrix(x_ref, x_ref, gamma)
    k_yy = _rbf_kernel_matrix(x_gen, x_gen, gamma)
    k_xy = _rbf_kernel_matrix(x_ref, x_gen, gamma)

    if estimator == "biased":
        mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    elif estimator == "unbiased":
        n_ref, n_gen = x_ref.shape[0], x_gen.shape[0]
        mmd2 = (k_xx.sum() - torch.trace(k_xx)) / (n_ref * (n_ref - 1))
        mmd2 = mmd2 + (k_yy.sum() - torch.trace(k_yy)) / (n_gen * (n_gen - 1))
        mmd2 = mmd2 - 2.0 * k_xy.mean()
    else:
        raise ValueError("estimator must be 'biased' or 'unbiased'.")

    return float(mmd2.item())


def sliced_wasserstein(x_ref, x_gen, n_projections=128, n_grid=1000, eps=1e-12, seed=None, n_quantiles=None):
    """Compute the sliced squared 2-Wasserstein distance."""
    n_grid = int(n_grid if n_quantiles is None else n_quantiles)
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)
    if int(n_projections) < 1:
        raise ValueError("n_projections must be at least 1.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    d = x_ref.shape[1]
    generator = None if seed is None else torch.Generator(device=x_ref.device).manual_seed(seed)
    dirs = torch.randn(int(n_projections), d, device=x_ref.device, dtype=x_ref.dtype, generator=generator)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=1, keepdim=True).clamp_min(torch.finfo(x_ref.dtype).eps)
    proj_ref = x_ref @ dirs.T
    proj_gen = x_gen @ dirs.T
    q = torch.linspace(0.0, 1.0 - float(eps), n_grid, device=x_ref.device, dtype=x_ref.dtype)
    q_ref = torch.quantile(proj_ref, q, dim=0)
    q_gen = torch.quantile(proj_gen, q, dim=0)
    return float(torch.trapz((q_ref - q_gen).pow(2), q, dim=0).mean().item())


def tail_coverage_curve(x_ref, x_gen, probs=None, tail="upper", min_exceedances=10):
    """Tail exceedance calibration at reference thresholds."""
    x_ref = _to_2d_tensor(x_ref)
    x_gen = _to_2d_tensor(x_gen, device=x_ref.device, dtype=x_ref.dtype)
    _validate_same_feature_dim(x_ref, x_gen)

    if probs is None:
        probs = _default_tail_probs(min(int(x_ref.shape[0]), int(x_gen.shape[0])), device=x_ref.device,
                                    dtype=x_ref.dtype, min_exceedances=min_exceedances)
    else:
        probs = _to_2d_tensor(probs, device=x_ref.device, dtype=x_ref.dtype).reshape(-1)
        if torch.any((probs <= 0) | (probs >= 1)):
            raise ValueError("probs must satisfy 0 < probs < 1")

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")

    quantiles = 1.0 - probs if tail == "upper" else probs
    thresholds = torch.quantile(x_ref, quantiles, dim=0)
    if tail == "upper":
        ref_coverage = (x_ref.unsqueeze(0) > thresholds.unsqueeze(1)).double().mean(dim=1)
        gen_coverage = (x_gen.unsqueeze(0) > thresholds.unsqueeze(1)).double().mean(dim=1)
    else:
        ref_coverage = (x_ref.unsqueeze(0) < thresholds.unsqueeze(1)).double().mean(dim=1)
        gen_coverage = (x_gen.unsqueeze(0) < thresholds.unsqueeze(1)).double().mean(dim=1)

    return probs, ref_coverage, gen_coverage


def tail_coverage_error(x_ref, x_gen, probs=None, tail="upper", min_exceedances=10, mode="log", reduction="mean", eps=1e-12):
    """Aggregate marginal tail-coverage mismatch over tail probabilities and features."""
    _, ref_cov, gen_cov = tail_coverage_curve(x_ref, x_gen, probs=probs, tail=tail,
                                              min_exceedances=min_exceedances)

    if mode == "log":
        err = (torch.log(gen_cov + float(eps)) - torch.log(ref_cov + float(eps))).abs()
    elif mode == "abs":
        err = (gen_cov - ref_cov).abs()
    else:
        raise ValueError("mode must be 'log' or 'abs'")

    if reduction == "mean":
        return float(err.mean().item())
    if reduction == "none":
        return err
    raise ValueError("reduction must be 'mean' or 'none'")
