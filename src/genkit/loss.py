"""Specific loss definitions."""

import math
import torch


def barron_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    scale: float | torch.Tensor = 1.0,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Barron robust loss (A General and Adaptive Robust Loss Function, Eq. 8).

    rho(x, alpha, c) =
        0.5 * (x / c)^2                                 if alpha == 2
        log(0.5 * (x / c)^2 + 1)                        if alpha == 0
        1 - exp(-0.5 * (x / c)^2)                       if alpha == -inf
        |alpha-2|/alpha * (((x/c)^2 / |alpha-2| + 1)^(alpha/2) - 1)  otherwise
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Invalid reduction: {reduction}")

    x = pred - target
    dtype = x.dtype
    device = x.device

    alpha = torch.as_tensor(alpha, dtype=dtype, device=device)
    scale = torch.as_tensor(scale, dtype=dtype, device=device).clamp_min(eps)

    z = (x / scale) ** 2

    loss_two = 0.5 * z
    loss_zero = torch.log1p(0.5 * z)
    loss_neginf = -torch.expm1(-0.5 * z)

    beta = torch.abs(alpha - 2.0).clamp_min(eps)
    loss_general = (beta / alpha) * (torch.pow(z / beta + 1.0, 0.5 * alpha) - 1.0)

    if alpha.ndim == 0:
        a = float(alpha.item())
        if math.isinf(a) and a < 0:
            loss = loss_neginf
        elif abs(a - 2.0) < 1e-6:
            loss = loss_two
        elif abs(a) < 1e-6:
            loss = loss_zero
        else:
            loss = loss_general
    else:
        is_two = torch.isclose(alpha, torch.tensor(2.0, dtype=dtype, device=device), atol=1e-6, rtol=0.0)
        is_zero = torch.isclose(alpha, torch.tensor(0.0, dtype=dtype, device=device), atol=1e-6, rtol=0.0)
        is_neginf = torch.isneginf(alpha)

        loss = torch.where(is_two, loss_two, loss_general)
        loss = torch.where(is_zero, loss_zero, loss)
        loss = torch.where(is_neginf, loss_neginf, loss)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
