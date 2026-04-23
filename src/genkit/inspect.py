"""Post-hoc inspection utilities."""

import contextlib
import io
import math
import warnings
from typing import Literal
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from ._abs import DDPMAbstarct
from .diffusion import DLPMEps, DDPMV, DDPMX0
from .flow import GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT
__all__ = [
    "model_est_err_curve",
    "model_est_jacobian_spectral_curve",
    "fit_hmm_on_weight_stats",
    "estimate_init_error",
    "estimate_training_loss_error",
]


def _require_family(gen_model: object) -> str:
    """Return the model family after validating it is inspectable."""
    family = getattr(gen_model, "_family", None)
    if family is None:
        raise ValueError(f"'inspect' utilities require a '._family' tag, got {type(gen_model)}.")
    if family not in {"flow", "diffusion"}:
        raise ValueError(f"'inspect' utilities can only inspect 'diffusion' or 'flow' models, got {family}.")
    return family


def _time_grid(gen_model: object, x: torch.Tensor, max_n_steps: int | None = None) -> torch.Tensor:
    """Return the sampled step grid used for Jacobian inspection."""
    family = _require_family(gen_model)
    if max_n_steps is not None and int(max_n_steps) < 1:
        raise ValueError("max_n_steps must be at least 1.")
    if family == "flow":
        t_grid = torch.arange(gen_model._n_steps, device=x.device, dtype=gen_model._idtype)
    else:
        t_grid = torch.arange(1, gen_model._n_steps + 1, device=x.device, dtype=gen_model._idtype)
    if max_n_steps is None or int(max_n_steps) >= len(t_grid):
        return t_grid
    idx = torch.linspace(0, len(t_grid) - 1, steps=int(max_n_steps), device=t_grid.device)
    return t_grid.index_select(0, torch.round(idx).to(dtype=torch.long))


def _jacobian_spectral_at_time(
    gen_model: object,
    x_eval: torch.Tensor,
    t_step: torch.Tensor,
    *,
    n_power_iter: int,
) -> torch.Tensor:
    """Estimate the maximum samplewise Jacobian spectral norm at one time."""
    family = _require_family(gen_model)
    denom = float(max(gen_model._n_steps - 1, 1)) if family == "flow" else float(gen_model._n_steps)
    t_batch = (t_step.to(dtype=gen_model._fdtype) / denom).reshape(1, 1)

    def model_at_time(z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(0)
        t = t_batch.to(device=z.device, dtype=z.dtype)
        return gen_model._net(x, t).squeeze(0)

    sample_values = []
    for x_i in x_eval:
        v = torch.randn_like(x_i)
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)

        for _ in range(int(n_power_iter)):
            x_base = x_i.detach()
            x_var = x_base.requires_grad_(True)
            y = model_at_time(x_var)
            _, jv = torch.autograd.functional.jvp(model_at_time, x_base, v, create_graph=False)
            jt_j_v = torch.autograd.grad(y, x_var, grad_outputs=jv, retain_graph=False, create_graph=False)[0]
            v = jt_j_v / torch.linalg.vector_norm(jt_j_v).clamp_min(1e-12)

        _, jv = torch.autograd.functional.jvp(model_at_time, x_i.detach(), v, create_graph=False)
        sample_values.append(torch.linalg.vector_norm(jv))

    return torch.stack(sample_values).max()


def _sample_forward_marginal(model, x_data: torch.Tensor, n_samples: int, t=None) -> torch.Tensor:
    """Draw samples from the model forward marginal at time t."""
    idx = torch.randint(x_data.shape[0], (n_samples,), device=x_data.device)
    x1 = x_data.index_select(0, idx).to(device=model._device, dtype=model._fdtype)

    if isinstance(model, DDPMAbstarct):
        t = model._n_steps if t is None else t
        _, x_t, _, _, _, _ = model._latent(x_1=x1, t=t)
        return x_t

    if isinstance(model, DLPMEps):
        t = model._n_steps if t is None else t
        t = model._check_t(t, n_samples).to(dtype=model._idtype)
        eps = model._sample_source_default(n_samples)
        gamma = model._gamma_1_t.index_select(0, t).unsqueeze(-1)
        sigma = model._sigma_1_t.index_select(0, t).unsqueeze(-1)
        return gamma * x1 + sigma * eps

    if isinstance(model, GaussianFlowLinear):
        t = model._t_min if t is None else float(t)
        x0 = model._sample_source(n_samples)
        t_vec = torch.full((n_samples, 1), t, device=model._device, dtype=model._fdtype)
        return (1.0 - t_vec) * x0 + t_vec * x1

    if isinstance(model, GaussianFlowOT):
        t = model._t_min if t is None else float(t)
        x0 = model._sample_source(n_samples)
        t_vec = torch.full((n_samples, 1), t, device=model._device, dtype=model._fdtype)
        return (1.0 - (1.0 - model._sigma_min) * t_vec) * x0 + t_vec * x1

    if isinstance(model, GaussianFlowDDPM):
        t = model._t_min if t is None else float(t)
        x0 = model._sample_source(n_samples)
        t_vec = torch.full((n_samples, 1), t, device=model._device, dtype=model._fdtype)
        _, _, a, sigma = model._vp_coefs(t_vec)
        return a * x1 + sigma * x0

    raise ValueError(
        f"Cannot build forward samples automatically for {type(model).__name__}. "
        "Pass `reference_sampler=` explicitly."
    )


def _knn_kl(p: torch.Tensor, q: torch.Tensor, k: int = 5) -> float:
    """Estimate KL(P || Q) from samples using the kNN estimator."""
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError("Expected flat tensors of shape (n_samples, dim).")
    if p.shape[1] != q.shape[1]:
        raise ValueError(f"Dimension mismatch: {p.shape[1]} vs {q.shape[1]}.")
    if p.shape[0] < 2 or q.shape[0] < 2:
        raise ValueError("Need at least two samples from each distribution.")

    k = min(int(k), p.shape[0] - 1, q.shape[0] - 1)
    if k < 1:
        raise ValueError("`k` must be >= 1.")

    eps = torch.finfo(p.dtype).eps
    d_pp = torch.cdist(p, p)
    d_pp.fill_diagonal_(float("inf"))
    rho = d_pp.kthvalue(k, dim=1).values.clamp_min(eps)
    d_pq = torch.cdist(p, q)
    nu = d_pq.kthvalue(k, dim=1).values.clamp_min(eps)
    kl = p.shape[1] * (nu.log() - rho.log()).mean() + math.log(q.shape[0] / (p.shape[0] - 1))
    return float(kl.item())


def _mse_loss_at_batch(model, x: torch.Tensor, t=None) -> torch.Tensor:
    """Compute a plain prediction-target MSE for one native training batch."""
    if isinstance(model, (GaussianFlowLinear, GaussianFlowOT, GaussianFlowDDPM)):
        pred, target, _ = model._precompute_loss(x=x, z=None, t=t)
        return torch.nn.functional.mse_loss(pred, target)

    if isinstance(model, DDPMV):
        x_1, x_t, eps, t_norm, _, a_bar_t = model._latent(x_1=x, eps=None, t=t)
        target = torch.sqrt(a_bar_t) * eps - torch.sqrt(1.0 - a_bar_t) * x_1
        pred = model._net(x_t, t_norm)
        return torch.nn.functional.mse_loss(pred, target)

    if isinstance(model, DDPMX0):
        x_1, x_t, _, t_norm, _, _ = model._latent(x_1=x, eps=None, t=t)
        pred = model._net(x_t, t_norm)
        return torch.nn.functional.mse_loss(pred, x_1)

    if isinstance(model, DLPMEps):
        x_1 = x.to(device=model._device, dtype=model._fdtype)
        n = x_1.size(0)
        n_a = model._n_trial_A
        n_g = model._n_trial_G
        t_checked = model._check_t(t, n) if t is not None else torch.randint(
            1, model._n_steps, (n,), device=model._device, dtype=model._idtype
        )
        t_e = t_checked.view(1, 1, n).expand(n_a, n_g, n).reshape(-1)
        t_norm = (t_checked / model._n_steps).view(1, 1, n).expand(n_a, n_g, n).reshape(-1, 1)
        eps = model._sample_source_default(n, expand_trials=True)
        gamma_1_t = model._gamma_1_t.index_select(0, t_e).unsqueeze(-1)
        sigma_1_t = model._sigma_1_t.index_select(0, t_e).unsqueeze(-1)
        x_1_e = x_1.view(1, 1, n, model._dim).expand(n_a, n_g, n, model._dim).reshape(-1, model._dim)
        x_t = gamma_1_t * x_1_e + sigma_1_t * eps
        pred = model._net(x_t, t_norm)
        return torch.nn.functional.mse_loss(pred, eps)

    raise ValueError(f"MSE inspection loss is not implemented for {type(model).__name__}.")


@torch.no_grad()
def model_est_err_curve(
    gen_model: object,
    x: torch.Tensor,
    *,
    max_n_steps: int | None = 10,
    loss_type: Literal["native", "mse"] = "native",
) -> np.ndarray:
    """Evaluate the mean score/vector fields estimation error across times."""
    family = _require_family(gen_model)
    if loss_type not in {"native", "mse"}:
        raise ValueError(f"loss_type must be 'native' or 'mse', got {loss_type!r}.")

    x_target = x.to(device=gen_model._device, dtype=gen_model._fdtype)
    x_source = gen_model._sample_source(len(x_target))
    t_grid = _time_grid(gen_model, x_target, max_n_steps=max_n_steps)
    was_training = gen_model._net.training
    gen_model._net.eval()

    if family == 'flow':
        t_grid = t_grid.float() / gen_model._n_steps

    try:
        if loss_type == "native":
            curve = torch.stack([gen_model.loss(x=x_target, z=x_source, t=t).detach() for t in t_grid])
        else:
            curve = torch.stack([_mse_loss_at_batch(gen_model, x_target, t=t).detach() for t in t_grid])
    finally:
        if was_training:
            gen_model._net.train()

    return curve.cpu().numpy()


def model_est_jacobian_spectral_curve(
    gen_model: object,
    x: torch.Tensor,
    *,
    max_n_steps: int | None = 10,
    n_power_iter: int = 8,
) -> np.ndarray:
    """Estimate the Jacobian spectral norm across times."""
    _ = _require_family(gen_model)
    if int(n_power_iter) < 1:
        raise ValueError("n_power_iter must be at least 1.")

    x_eval = x.to(device=gen_model._device, dtype=gen_model._fdtype)
    t_grid = _time_grid(gen_model, x_eval, max_n_steps=max_n_steps)
    was_training = gen_model._net.training
    gen_model._net.eval()

    try:
        curve = torch.stack([_jacobian_spectral_at_time(gen_model, x_eval, t_step, n_power_iter=n_power_iter) for t_step in t_grid])
    finally:
        if was_training:
            gen_model._net.train()

    return curve.cpu().numpy()


def fit_hmm_on_weight_stats(weight_stats, n_states=3, random_state=0):
    """Fit an HMM on per-epoch weight statistics to segment training phases."""
    X = np.asarray(weight_stats, dtype=float)
    if X.ndim != 2:
        raise ValueError("weight_stats must be a 2D array-like object of shape (n_epochs, n_stats).")
    if X.shape[1] == 0:
        raise ValueError("weight_stats must contain at least one statistic per epoch.")

    Z = StandardScaler().fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            hmm = GaussianHMM(n_components=int(n_states), covariance_type="diag", n_iter=200, random_state=random_state).fit(Z)

    return {"stats": X,
            "stat_names": ["mean_l2", "mean_top_singular", "mean_weight", "mean_std"] if X.shape[1] == 4 else [f"stat_{i}" for i in range(X.shape[1])],
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
    _require_family(model)
    if source_sampler is None:
        if not hasattr(model, "_sample_source"):
            raise ValueError("Pass `source_sampler=` because this model has no `_sample_source` method.")
        source_sampler = model._sample_source

    p = reference_sampler(n_samples) if reference_sampler is not None else _sample_forward_marginal(model, x_data, n_samples, t=t)
    q = source_sampler(n_samples)
    p = p.to(device=model._device, dtype=model._fdtype).reshape(n_samples, -1)
    q = q.to(device=model._device, dtype=model._fdtype).reshape(n_samples, -1)
    return _knn_kl(p, q, k=k)


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
    _require_family(model)
    if loss_type not in {"native", "mse"}:
        raise ValueError(f"loss_type must be 'native' or 'mse', got {loss_type!r}.")

    was_training = model._net.training
    model._net.eval()
    try:
        values = []
        for _ in range(int(n_batches)):
            idx = torch.randint(x_data.shape[0], (int(batch_size),), device=model._device)
            x = x_data.index_select(0, idx).to(device=model._device, dtype=model._fdtype)
            if loss_type == "native":
                values.append(model.loss(x, t=t).detach())
            else:
                values.append(_mse_loss_at_batch(model, x, t=t).detach())
        return float(torch.stack(values).mean().item())
    finally:
        if was_training:
            model._net.train()
