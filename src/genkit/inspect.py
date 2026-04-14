"""Post-hoc inspection utilities."""

from typing import Literal
from typing import Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ._inspect import (
    err_time_grid,
    jacobian_spectral_at_time,
    jacobian_step_grid,
    knn_kl,
    mse_loss_at_batch,
    require_family,
    sample_forward_marginal,
    subsample_rows,
)


@torch.no_grad()
def model_est_err_curve(
    gen_model: object,
    x: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the model loss across its native time grid."""
    require_family(gen_model)
    x_target = x
    x_source = gen_model._sample_source(len(x_target))
    t_grid = err_time_grid(gen_model, x)
    was_training = gen_model._net.training
    gen_model._net.eval()
    try:
        loss_values = [gen_model._loss(x=x_target, z=x_source, t=t) for t in t_grid]
        return torch.stack(loss_values).cpu().numpy(), t_grid.cpu().numpy()
    finally:
        if was_training:
            gen_model._net.train()


def model_est_jacobian_spectral_curve(
    gen_model: object,
    x: torch.Tensor,
    *,
    n_power_iter: int = 8,
    max_n_steps: int | None = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the Jacobian spectral norm across sampled model times."""
    if n_power_iter < 1:
        raise ValueError("n_power_iter must be at least 1.")
    require_family(gen_model)
    x_eval = x.to(device=x.device, dtype=gen_model._fdtype)
    t_grid = jacobian_step_grid(gen_model, x_eval, max_n_steps)
    was_training = gen_model._net.training
    gen_model._net.eval()
    try:
        curve = [jacobian_spectral_at_time(gen_model, x_eval, t_step, n_power_iter=n_power_iter) for t_step in t_grid]
    finally:
        if was_training:
            gen_model._net.train()

    return torch.stack(curve).cpu().numpy(), t_grid.cpu().numpy()


def fit_hmm_on_weight_stats(weight_stats, n_states=None, random_state=0):
    """Fit an HMM on per-epoch weight statistics to segment training phases."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("hmmlearn is required for fit_hmm_on_weight_stats.") from exc

    X = np.asarray(weight_stats, dtype=float)
    if X.ndim != 2:
        raise ValueError("weight_stats must be a 2D array-like object of shape (n_epochs, n_stats).")
    if X.shape[0] < 3:
        raise ValueError("weight_stats must contain at least 3 epochs.")
    if X.shape[1] == 0:
        raise ValueError("weight_stats must contain at least one statistic per epoch.")

    Z = StandardScaler().fit_transform(X)
    if n_states is None:
        max_states = min(6, X.shape[0] - 1)
        candidates = range(2, max_states + 1)
        scored_models = []
        for k in candidates:
            hmm = GaussianHMM(
                n_components=k,
                covariance_type="diag",
                n_iter=200,
                random_state=random_state,
            ).fit(Z)
            scored_models.append((hmm.bic(Z), hmm))
        _, hmm = min(scored_models, key=lambda item: item[0])
    else:
        hmm = GaussianHMM(
            n_components=int(n_states),
            covariance_type="diag",
            n_iter=200,
            random_state=random_state,
        ).fit(Z)

    return {
        "stats": X,
        "stat_names": (
            ["mean_l2", "mean_top_singular", "mean_weight", "mean_std"]
            if X.shape[1] == 4
            else [f"stat_{i}" for i in range(X.shape[1])]
        ),
        "states": hmm.predict(Z),
        "state_probs": hmm.predict_proba(Z),
        "n_states": int(hmm.n_components),
        "hmm": hmm,
    }


@torch.no_grad()
def estimate_init_error(
    model,
    x_data: torch.Tensor,
    *,
    t=None,
    n_samples: int = 1024,
    k: int = 5,
    source_sampler=None,
    reference_sampler=None,
) -> float:
    """Estimate the initialization error KL(p_t || q_init)."""
    require_family(model)
    if source_sampler is None:
        if not hasattr(model, "_sample_source"):
            raise ValueError("Pass `source_sampler=` because this model has no `_sample_source` method.")
        source_sampler = model._sample_source

    p = reference_sampler(n_samples) if reference_sampler is not None else sample_forward_marginal(
        model, x_data, n_samples, t=t
    )
    q = source_sampler(n_samples)

    p = p.to(device=model._device, dtype=model._fdtype).reshape(n_samples, -1)
    q = q.to(device=model._device, dtype=model._fdtype).reshape(n_samples, -1)
    return knn_kl(p, q, k=k)


@torch.no_grad()
def estimate_training_loss_error(
    model,
    x_data: torch.Tensor,
    *,
    t=None,
    n_batches: int = 32,
    batch_size: int = 256,
    loss_type: Literal["native", "mse"] = "native",
) -> float:
    """Monte Carlo estimate of the native or plain-MSE training loss on a reference split."""
    require_family(model)
    if loss_type not in {"native", "mse"}:
        raise ValueError(f"loss_type must be 'native' or 'mse', got {loss_type!r}.")
    was_training = model._net.training
    model._net.eval()
    try:
        values = []
        for _ in range(int(n_batches)):
            x = subsample_rows(x_data, int(batch_size), device=model._device, dtype=model._fdtype)
            if loss_type == "native":
                values.append(model.loss(x, t=t).detach())
            else:
                values.append(mse_loss_at_batch(model, x, t=t).detach())
        return float(torch.stack(values).mean().item())
    finally:
        if was_training:
            model._net.train()


def estimate_training_error(*args, **kwargs) -> float:
    """Backward-compatible alias for :func:`estimate_training_loss_error`."""
    return estimate_training_loss_error(*args, **kwargs)
