"""Post-hoc inspection utilities."""

from typing import Tuple
import numpy as np
import torch


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


@torch.no_grad()
def model_est_err_curve(
    gen_model: object,
    x: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the model loss across its native time grid."""
    x_target = x
    x_source = gen_model._sample_source(len(x_target))

    if not hasattr(gen_model, "_family"):
        raise ValueError(f"'model_estimation_error_curve' can't assess models without a '._family' tag: {type(gen_model)}.")
    if gen_model._family == "flow":
        t_grid = torch.linspace(0.0, 1.0, steps=gen_model._n_steps, device=x.device, dtype=gen_model._fdtype)
    elif gen_model._family == "diffusion":
        t_grid = torch.arange(1, gen_model._n_steps + 1, device=x.device, dtype=gen_model._idtype)
    else:
        raise ValueError(
            f"'model_estimation_error_curve' can only inspect 'diffusion' or 'flow' models, got {gen_model._family}."
        )

    was_training = gen_model._net.training
    gen_model._net.eval()

    loss_values = []
    for t in t_grid:
        loss_values.append(gen_model._loss(x=x_target, z=x_source, t=t))

    if was_training:
        gen_model._net.train()

    return torch.stack(loss_values).cpu().numpy(), t_grid.cpu().numpy()
