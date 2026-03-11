"""Classical evaluation metrics."""

from typing import Optional
import torch

__all__ = [
    "sliced_wasserstein2",
    "msle",
    "msle_90",
    "msle_99",
    "mse",
    "rnmse",
    "mae",
]


def _as_pair(x_true: torch.Tensor, x_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(x_true, torch.Tensor) or not isinstance(x_pred, torch.Tensor):
        raise TypeError("x_true and x_pred must be torch.Tensor.")
    x_true = x_true.detach()
    x_pred = x_pred.detach()
    if x_pred.device != x_true.device:
        x_pred = x_pred.to(device=x_true.device)
    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(x_true.shape)} vs {tuple(x_pred.shape)}")
    return x_true, x_pred


def _out_dtype(x: torch.Tensor) -> torch.dtype:
    return x.dtype if x.dtype.is_floating_point else torch.float32


@torch.no_grad()
def mse(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    x_true, x_pred = _as_pair(x_true, x_pred)
    return ((x_true - x_pred) ** 2).mean().to(dtype=_out_dtype(x_true))


@torch.no_grad()
def mae(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    x_true, x_pred = _as_pair(x_true, x_pred)
    return (x_true - x_pred).abs().mean().to(dtype=_out_dtype(x_true))


@torch.no_grad()
def rnmse(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x_true, x_pred = _as_pair(x_true, x_pred)
    rmse = torch.sqrt(((x_true - x_pred) ** 2).mean())
    denom = torch.sqrt((x_true ** 2).mean()).clamp_min(float(eps))
    return (rmse / denom).to(dtype=_out_dtype(x_true))


def _msle_at_quantile(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    xi: float,
    n_grid: int = 256,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not (0.0 < xi < 1.0):
        raise ValueError(f"xi must be in (0,1), got {xi}.")
    if n_grid <= 0:
        raise ValueError("n_grid must be positive.")

    x_true, x_pred = _as_pair(x_true, x_pred)
    a = x_true.reshape(-1)
    b = x_pred.reshape(-1)

    p = torch.linspace(xi, 1.0, n_grid, device=a.device, dtype=a.dtype)
    qa = torch.quantile(a, p)
    qb = torch.quantile(b, p)
    diff2 = (torch.log(qa.clamp_min(eps)) - torch.log(qb.clamp_min(eps))) ** 2

    if n_grid == 1:
        out = (1.0 - xi) * diff2[0]
    else:
        dp = (1.0 - xi) / float(n_grid - 1)
        out = diff2.sum() * dp
    return out.to(dtype=_out_dtype(x_true))


@torch.no_grad()
def msle(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    x_true, x_pred = _as_pair(x_true, x_pred)
    a = x_true.reshape(-1)
    b = x_pred.reshape(-1)
    out = (torch.log(a.clamp_min(eps)) - torch.log(b.clamp_min(eps))) ** 2
    return out.mean().to(dtype=_out_dtype(x_true))


@torch.no_grad()
def msle_90(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    n_grid: int = 256,
    eps: float = 1e-12,
) -> torch.Tensor:
    return _msle_at_quantile(x_true=x_true, x_pred=x_pred, xi=0.90, n_grid=n_grid, eps=eps)


@torch.no_grad()
def msle_99(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    n_grid: int = 256,
    eps: float = 1e-12,
) -> torch.Tensor:
    return _msle_at_quantile(x_true=x_true, x_pred=x_pred, xi=0.99, n_grid=n_grid, eps=eps)


@torch.no_grad()
def sliced_wasserstein2(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 128,
    sqrt: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    x, y = _as_pair(x, y)
    if x.ndim != 2:
        raise ValueError("x and y must be 2D tensors with shape (n, d).")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive.")

    n, d = x.shape
    if y.shape[0] != n:
        raise ValueError(f"Use same #samples for SW2, got {n} vs {y.shape[0]}.")

    gen = None
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))

    theta = torch.randn((n_projections, d), device=x.device, dtype=x.dtype, generator=gen)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)

    x_proj = x @ theta.T
    y_proj = y @ theta.T
    x_sort = torch.sort(x_proj, dim=0).values
    y_sort = torch.sort(y_proj, dim=0).values

    w2_sq = ((x_sort - y_sort) ** 2).mean()
    out = torch.sqrt(w2_sq) if sqrt else w2_sq
    return out.to(dtype=_out_dtype(x))
