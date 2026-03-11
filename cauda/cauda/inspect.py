"""Post-hoc inspection utilities."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch

__all__ = [
    "nearest_train_sample",
    "nearest_neighbor_generalization_stats",
    "inspect_gaussian_linear_flow",
]


def _as_pair(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be torch.Tensor.")
    x = x.detach()
    y = y.detach()
    if y.device != x.device:
        y = y.to(device=x.device)
    return x, y


def _summary_from_tensor(values: torch.Tensor) -> Dict[str, float]:
    t = values.reshape(-1).to(dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64, device=t.device))
    return {
        "nn_dist_mean": float(t.mean().item()),
        "nn_dist_std": float(t.std(unbiased=False).item()),
        "nn_dist_median": float(q[0].item()),
        "nn_dist_q90": float(q[1].item()),
        "nn_dist_q95": float(q[2].item()),
        "nn_dist_q99": float(q[3].item()),
        "nn_dist_min": float(t.min().item()),
        "nn_dist_max": float(t.max().item()),
    }


@torch.no_grad()
def nearest_train_sample(
    x_gen: torch.Tensor,
    x_train: torch.Tensor,
    metric: str = "cosine",
    return_distance: bool = True,
    block_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    x_gen, x_train = _as_pair(x_gen, x_train)
    if x_gen.ndim != 2 or x_train.ndim != 2:
        raise ValueError("x_gen and x_train must be 2D tensors with shape (n, d).")
    if x_gen.shape[1] != x_train.shape[1]:
        raise ValueError(f"Dim mismatch: {x_gen.shape[1]} vs {x_train.shape[1]}")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    metric = str(metric).lower()
    if metric not in ("cosine", "l2", "l1"):
        raise ValueError(f"metric must be one of ['cosine','l2','l1'], got '{metric}'.")

    n = x_gen.shape[0]
    out_dtype = x_gen.dtype if x_gen.dtype.is_floating_point else torch.float32

    if metric == "cosine":
        eps = torch.finfo(out_dtype).tiny
        g = x_gen / x_gen.norm(dim=1, keepdim=True).clamp_min(eps)
        t = x_train / x_train.norm(dim=1, keepdim=True).clamp_min(eps)

        best_score = torch.full((n,), -float("inf"), device=x_gen.device, dtype=torch.float64)
        best_index = torch.zeros((n,), device=x_gen.device, dtype=torch.long)

        for i in range(0, n, block_size):
            gi = g[i: i + block_size]
            val, idx = torch.max(gi @ t.T, dim=1)
            b = gi.size(0)
            best_score[i: i + b] = val.to(dtype=torch.float64)
            best_index[i: i + b] = idx

        nn_sample = x_train.index_select(0, best_index)
        if not return_distance:
            return best_index, nn_sample, None
        nn_distance = (1.0 - best_score).clamp_min(0.0).to(dtype=out_dtype)
        return best_index, nn_sample, nn_distance

    best_dist = torch.full((n,), float("inf"), device=x_gen.device, dtype=torch.float64)
    best_index = torch.zeros((n,), device=x_gen.device, dtype=torch.long)

    for i in range(0, n, block_size):
        gi = x_gen[i: i + block_size]
        dmat = torch.cdist(gi, x_train, p=2).pow(2) if metric == "l2" else torch.cdist(gi, x_train, p=1)
        val, idx = torch.min(dmat, dim=1)
        b = gi.size(0)
        best_dist[i: i + b] = val.to(dtype=torch.float64)
        best_index[i: i + b] = idx

    nn_sample = x_train.index_select(0, best_index)
    if not return_distance:
        return best_index, nn_sample, None
    return best_index, nn_sample, best_dist.to(dtype=out_dtype)


@torch.no_grad()
def nearest_neighbor_generalization_stats(
    x_gen: torch.Tensor,
    x_train: torch.Tensor,
    metric: str = "cosine",
    block_size: int = 1024,
) -> Dict[str, float]:
    _, _, nn_dist = nearest_train_sample(
        x_gen=x_gen,
        x_train=x_train,
        metric=metric,
        return_distance=True,
        block_size=block_size,
    )
    assert nn_dist is not None
    return _summary_from_tensor(nn_dist)


@torch.no_grad()
def inspect_gaussian_linear_flow(gen_model, x: torch.Tensor, steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
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
