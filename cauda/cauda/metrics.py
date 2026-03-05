"""Metrics module."""

from typing import Optional
import torch


__all__ = [
    "msle_at_quantile",
    "fid",
    "wasserstein2_1d",
    "sliced_wasserstein2",
    "mmd_rbf",
    "mmd_imq",
    "f1",
]


def _check_2d(x: torch.Tensor, y: torch.Tensor) -> tuple[int, int, int]:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors with shape (n, d).")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Dim mismatch: x has d={x.shape[1]} but y has d={y.shape[1]}.")
    n, d = x.shape
    m = y.shape[0]
    return n, m, d


def _linalg_work_dtype(x: torch.Tensor) -> torch.dtype:
    if x.device.type == "mps":
        return torch.float32
    return torch.float64


@torch.no_grad()
def f1(
    x_true: torch.Tensor,
    x_gen: torch.Tensor,
    threshold: float = 0.5,
    zero_division: float = 0.0,
) -> torch.Tensor:
    """
    F1 score for a binary mask inferred from x_true and x_gen.
    """
    if x_true.shape != x_gen.shape:
        raise ValueError(f"Shape mismatch: x_true.shape={tuple(x_true.shape)}"
                         f", x_gen.shape={tuple(x_gen.shape)}")

    if x_true.dtype.is_floating_point:
        x_bin = x_true >= threshold
    else:
        x_bin = x_true != 0

    if x_gen.dtype.is_floating_point:
        y_bin = x_gen >= threshold
    else:
        y_bin = x_gen != 0

    tp = (x_bin & y_bin).sum(dtype=torch.float32)
    fp = ((~x_bin) & y_bin).sum(dtype=torch.float32)
    fn = (x_bin & (~y_bin)).sum(dtype=torch.float32)

    denom = 2.0 * tp + fp + fn
    z = torch.tensor(zero_division, device=x_true.device, dtype=torch.float32)
    return torch.where(denom > 0, (2.0 * tp) / denom, z)


@torch.no_grad()
def msle_at_quantile(
    x_true: torch.Tensor,
    x_gen: torch.Tensor,
    xi: float = 0.95,
    n_grid: int = 256,
    eps: float = 1e-12,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Tail MSLE(xi) = Integral_{xi}^{1} (log Q_true(p) - log Q_gen(p))^2 dp
    using a Riemann sum on a uniform grid of p in [xi, 1].
    """
    if not (0.0 < xi < 1.0):
        raise ValueError(f"xi must be in (0,1), got {xi}.")
    if n_grid <= 0:
        raise ValueError(f"n_grid must be positive, got {n_grid}.")

    def _msle_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.reshape(-1)
        b = b.reshape(-1)
        p = torch.linspace(xi, 1.0, n_grid, device=a.device, dtype=a.dtype)
        qa = torch.quantile(a, p)
        qb = torch.quantile(b, p)
        da = torch.log(qa.clamp_min(eps))
        db = torch.log(qb.clamp_min(eps))
        diff2 = (da - db).pow(2)
        if n_grid == 1:
            return (1.0 - xi) * diff2[0]
        dp = (1.0 - xi) / float(n_grid - 1)
        return diff2.sum() * dp

    x_true = x_true.detach()
    x_gen = x_gen.detach()

    if x_true.ndim == 1:
        if x_gen.ndim != 1:
            raise ValueError(f"x_gen must be 1D if x_true is 1D, got {x_gen.ndim}D.")
        return _msle_1d(x_true, x_gen)

    if x_true.ndim == 2:
        if x_gen.ndim != 2 or x_gen.shape != x_true.shape:
            raise ValueError(f"Expected x_gen shape {tuple(x_true.shape)}, got {tuple(x_gen.shape)}.")
        if dim is not None:
            return _msle_1d(x_true[:, dim], x_gen[:, dim])
        vals = torch.stack([_msle_1d(x_true[:, j], x_gen[:, j]) for j in range(x_true.size(1))])
        return vals.mean()

    raise ValueError(f"x_true must be 1D or 2D, got shape {tuple(x_true.shape)}.")


@torch.no_grad()
def fid(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Fréchet Inception Distance (FID) between two sets of samples.
    For samples x ~ (N, D), y ~ (M, D):
        FID = ||μx - μy||^2 + Tr(Σx + Σy - 2 (Σx^{1/2} Σy Σx^{1/2})^{1/2})
    """
    n, m, d = _check_2d(x, y)
    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples per set to estimate covariances for FID.")

    work_dtype = _linalg_work_dtype(x)
    xw = x.to(dtype=work_dtype)
    yw = y.to(dtype=work_dtype)

    mu_x = xw.mean(dim=0)
    mu_y = yw.mean(dim=0)
    xc = xw - mu_x
    yc = yw - mu_y

    cov_x = (xc.T @ xc) / (n - 1)
    cov_y = (yc.T @ yc) / (m - 1)

    eye = torch.eye(d, device=x.device, dtype=work_dtype)
    cov_x = cov_x + float(eps) * eye
    cov_y = cov_y + float(eps) * eye

    ex, vx = torch.linalg.eigh(0.5 * (cov_x + cov_x.T))
    ex = ex.clamp_min(0.0)
    sqrt_cov_x = (vx * ex.sqrt().unsqueeze(0)) @ vx.T

    a = sqrt_cov_x @ cov_y @ sqrt_cov_x
    a = 0.5 * (a + a.T)

    ea, va = torch.linalg.eigh(a)
    ea = ea.clamp_min(0.0)
    sqrt_a = (va * ea.sqrt().unsqueeze(0)) @ va.T

    diff = mu_x - mu_y
    fid_val = diff.dot(diff) + torch.trace(cov_x) + torch.trace(cov_y) - 2.0 * torch.trace(sqrt_a)
    fid_val = fid_val.clamp_min(0.0)
    return fid_val.to(dtype=x.dtype)


@torch.no_grad()
def wasserstein2_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    sqrt: bool = True,
) -> torch.Tensor:
    """
    Empirical Wasserstein-2 distance for 1D with equal weights.
    For equal sample sizes n:
        W2^2 = (1/n) * sum_i (x_(i) - y_(i))^2
    where x_(i), y_(i) are sorted.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.numel() != y.numel():
        raise ValueError(f"Need same #samples for 1D W2, got {x.numel()} vs {y.numel()}.")

    xs = torch.sort(x).values
    ys = torch.sort(y).values
    w2_sq = torch.mean((xs - ys) ** 2)
    return torch.sqrt(w2_sq) if sqrt else w2_sq


@torch.no_grad()
def sliced_wasserstein2(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 128,
    sqrt: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sliced Wasserstein-2 (SW2) for d>1 using random 1D projections.
        SW2^2 = E_θ [ W2^2( <θ, x>, <θ, y> ) ],  θ ~ Uniform(S^{d-1})
    This uses Monte Carlo projections and 1D sorting. Requires equal sample sizes.
    """
    n, m, d = _check_2d(x, y)
    if n != m:
        raise ValueError(f"Use same #samples for SW2 (equal weights), got {n} vs {m}.")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive.")

    device, dtype = x.device, x.dtype
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    theta = torch.randn((n_projections, d), device=device, dtype=dtype, generator=gen)
    denom = theta.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    theta = theta / denom

    x_proj = x @ theta.T
    y_proj = y @ theta.T
    x_sort = torch.sort(x_proj, dim=0).values
    y_sort = torch.sort(y_proj, dim=0).values

    sw2_sq = torch.mean((x_sort - y_sort) ** 2)
    return torch.sqrt(sw2_sq) if sqrt else sw2_sq


def _median_heuristic_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """
    Median heuristic for RBF bandwidth from a subsample of pairwise distances.
    """
    _check_2d(x, y)
    device, dtype = x.device, x.dtype
    n = x.shape[0]
    m = min(int(max_points), n)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    idx_x = torch.randperm(n, device=device, generator=gen)[:m]
    idx_y = torch.randperm(y.shape[0], device=device, generator=gen)[: min(m, y.shape[0])]

    xs = x[idx_x]
    ys = y[idx_y]

    d = torch.cdist(xs, ys)
    d2 = d * d
    med = torch.median(d2)
    return torch.sqrt(med.clamp_min(torch.finfo(dtype).tiny))


@torch.no_grad()
def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    unbiased: bool = True,
    block_size: int = 1024,
    sigma_seed: int = 0,
) -> torch.Tensor:
    """
    MMD^2 with RBF kernel k(a,b)=exp(-||a-b||^2/(2*sigma^2)).
    Unbiased estimator:
        MMD^2 = 1/(n(n-1)) sum_{i!=j} k(x_i,x_j)
             + 1/(m(m-1)) sum_{i!=j} k(y_i,y_j)
             - 2/(nm)     sum_{i,j}  k(x_i,y_j)

    Computed blockwise to avoid materializing full pairwise matrices.
    """
    n, m, d = _check_2d(x, y)
    if unbiased and (n < 2 or m < 2):
        raise ValueError("Need at least 2 samples per set for unbiased MMD.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    device, dtype = x.device, x.dtype
    if sigma is None:
        sigma = _median_heuristic_sigma(x, y, seed=sigma_seed)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    sigma2 = (sigma * sigma).clamp_min(torch.finfo(dtype).tiny)
    inv_2sigma2 = 0.5 / sigma2

    def sum_rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float64)
        bt = b.T
        b2 = (b * b).sum(dim=1, keepdim=True).T
        a2 = (a * a).sum(dim=1, keepdim=True)

        for i in range(0, a.shape[0], block_size):
            ai = a[i: i + block_size]
            ai2 = a2[i: i + block_size]
            d2 = ai2 + b2 - 2.0 * (ai @ bt)
            d2 = d2.clamp_min(0.0)
            k = torch.exp(-d2 * inv_2sigma2)
            total += k.sum(dtype=torch.float64)
        return total

    k_xx = sum_rbf(x, x)
    k_yy = sum_rbf(y, y)
    k_xy = sum_rbf(x, y)

    if unbiased:
        k_xx = k_xx - float(n)
        k_yy = k_yy - float(m)
        mmd2 = k_xx / (n * (n - 1)) + k_yy / (m * (m - 1)) - 2.0 * k_xy / (n * m)
    else:
        mmd2 = k_xx / (n * n) + k_yy / (m * m) - 2.0 * k_xy / (n * m)

    return mmd2.to(dtype=dtype)


@torch.no_grad()
def mmd_imq(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    beta: float = 0.5,
    unbiased: bool = True,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    MMD^2 with IMQ kernel k(a,b) = (c / (c + ||a-b||^2))^beta.
    Often more stable than RBF for heavy-tailed settings.
    """
    n, m, _ = _check_2d(x, y)
    if unbiased and (n < 2 or m < 2):
        raise ValueError("Need at least 2 samples per set for unbiased MMD.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    device, dtype = x.device, x.dtype
    c_t = torch.as_tensor(c, device=device, dtype=dtype).clamp_min(torch.finfo(dtype).tiny)
    beta_f = float(beta)

    def sum_imq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float64)
        bt = b.T
        b2 = (b * b).sum(dim=1, keepdim=True).T
        a2 = (a * a).sum(dim=1, keepdim=True)

        for i in range(0, a.shape[0], block_size):
            ai = a[i: i + block_size]
            ai2 = a2[i: i + block_size]
            d2 = ai2 + b2 - 2.0 * (ai @ bt)
            d2 = d2.clamp_min(0.0)
            k = (c_t / (c_t + d2)) ** beta_f
            total += k.sum(dtype=torch.float64)
        return total

    k_xx = sum_imq(x, x)
    k_yy = sum_imq(y, y)
    k_xy = sum_imq(x, y)

    if unbiased:
        k_xx = k_xx - float(n)
        k_yy = k_yy - float(m)
        mmd2 = k_xx / (n * (n - 1)) + k_yy / (m * (m - 1)) - 2.0 * k_xy / (n * m)
    else:
        mmd2 = k_xx / (n * n) + k_yy / (m * m) - 2.0 * k_xy / (n * m)

    return mmd2.to(dtype=dtype)
