"""Post-hoc inspection utilities."""

from typing import Tuple
import numpy as np
import torch


@torch.no_grad()
def nn_dist_min(
    x_gen: torch.Tensor,
    x_train: torch.Tensor,
    metric: str = "sd",
    block_size: int = 1024,
) -> float:
    """Return the minimal generated-to-train nearest-neighbor distance."""
    if not isinstance(x_gen, torch.Tensor) or not isinstance(x_train, torch.Tensor):
        raise TypeError("x_gen and x_train must be torch.Tensor.")
    x_gen = x_gen.detach()
    x_train = x_train.detach()
    if x_train.device != x_gen.device:
        x_train = x_train.to(device=x_gen.device)

    if x_gen.ndim != 2 or x_train.ndim != 2:
        raise ValueError("x_gen and x_train must be 2D tensors with shape (n, d).")
    if x_gen.shape[1] != x_train.shape[1]:
        raise ValueError(f"Dim mismatch: {x_gen.shape[1]} vs {x_train.shape[1]}")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    metric = str(metric).lower()
    if metric not in ("sd", "l1"):
        raise ValueError(f"metric must be one of ['sd','l1'], got '{metric}'.")

    n = x_gen.shape[0]

    best_dist = torch.full((n,), float("inf"), device=x_gen.device, dtype=torch.float64)
    for i in range(0, n, block_size):
        gi = x_gen[i: i + block_size]
        dmat = torch.cdist(gi, x_train, p=2).pow(2) if metric == "sd" else torch.cdist(gi, x_train, p=1)
        val = torch.min(dmat, dim=1).values
        b = gi.size(0)
        best_dist[i: i + b] = val.to(dtype=torch.float64)
    return float(best_dist.min().item())


@torch.no_grad()
def linear_flow_velocity_mse_curve(
    gen_model, x: torch.Tensor,
    steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate mean-squared velocity error along the linear interpolation path from source to data."""
    x_1 = x
    x_0 = gen_model._sample_source(len(x_1))
    v_t = x_1 - x_0
    tt = torch.linspace(0.0, 1.0, steps=int(steps), device=x.device, dtype=x.dtype)

    was_training = gen_model._net.training
    gen_model._net.eval()

    err_t = []
    for t in tt:
        x_t = (1.0 - t) * x_0 + t * x_1
        t_batch = torch.full((len(x_1), 1), t.item(), device=x.device, dtype=x.dtype)
        v_t_hat = gen_model._net(x_t, t_batch)
        err_t.append(((v_t_hat - v_t) ** 2).mean())

    if was_training:
        gen_model._net.train()

    return torch.stack(err_t).cpu().numpy(), tt.cpu().numpy()
