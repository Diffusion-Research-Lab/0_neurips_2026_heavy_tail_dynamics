"""Private helpers for model inspection."""

import math
import torch
from ._abs import DDPMAbstarct
from .diffusion import DLPMEps
from .flow import GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT


def require_family(gen_model: object) -> str:
    """Return the model family after validating it is inspectable."""
    family = getattr(gen_model, "_family", None)
    if family is None:
        raise ValueError(f"'inspect' utilities require a '._family' tag, got {type(gen_model)}.")
    if family not in {"flow", "diffusion"}:
        raise ValueError(f"'inspect' utilities can only inspect 'diffusion' or 'flow' models, got {family}.")
    return family


def err_time_grid(gen_model: object, x: torch.Tensor) -> torch.Tensor:
    """Return the native loss-evaluation time grid."""
    family = require_family(gen_model)
    if family == "flow":
        return torch.linspace(0.0, 1.0, steps=gen_model._n_steps, device=x.device, dtype=gen_model._fdtype)
    return torch.arange(1, gen_model._n_steps + 1, device=x.device, dtype=gen_model._idtype)


def jacobian_step_grid(gen_model: object, x: torch.Tensor, max_n_steps: int | None) -> torch.Tensor:
    """Return the sampled step grid used for Jacobian inspection."""
    family = require_family(gen_model)
    if max_n_steps is not None and max_n_steps < 1:
        raise ValueError("max_n_steps must be at least 1 or None.")
    if family == "flow":
        t_grid = torch.arange(gen_model._n_steps, device=x.device, dtype=gen_model._idtype)
    else:
        t_grid = torch.arange(1, gen_model._n_steps + 1, device=x.device, dtype=gen_model._idtype)
    if max_n_steps is None or max_n_steps >= len(t_grid):
        return t_grid
    idx = torch.linspace(0, len(t_grid) - 1, steps=int(max_n_steps), device=t_grid.device)
    return t_grid.index_select(0, torch.round(idx).to(dtype=torch.long))


def jacobian_spectral_at_time(
    gen_model: object,
    x_eval: torch.Tensor,
    t_step: torch.Tensor,
    *,
    n_power_iter: int,
) -> torch.Tensor:
    """Estimate the maximum samplewise Jacobian spectral norm at one time."""
    family = require_family(gen_model)
    if family == "flow":
        denom = float(max(gen_model._n_steps - 1, 1))
    else:
        denom = float(gen_model._n_steps)
    t_batch = (t_step.to(dtype=gen_model._fdtype) / denom).reshape(1, 1)
    sample_values = []

    for x_i in x_eval:
        def model_at_time(z: torch.Tensor) -> torch.Tensor:
            return gen_model._net(z.unsqueeze(0), t_batch.to(device=z.device, dtype=z.dtype)).squeeze(0)

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


def subsample_rows(x: torch.Tensor, n: int, *, device, dtype) -> torch.Tensor:
    """Sample rows with replacement and cast to the requested device and dtype."""
    idx = torch.randint(x.shape[0], (n,), device=x.device)
    return x.index_select(0, idx).to(device=device, dtype=dtype)


def sample_forward_marginal(model, x_data: torch.Tensor, n_samples: int, t=None) -> torch.Tensor:
    """Draw samples from the model forward marginal at time t."""
    x1 = subsample_rows(x_data, n_samples, device=model._device, dtype=model._fdtype)

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


def knn_kl(p: torch.Tensor, q: torch.Tensor, k: int = 5) -> float:
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

    d = p.shape[1]
    eps = torch.finfo(p.dtype).eps
    d_pp = torch.cdist(p, p)
    d_pp.fill_diagonal_(float("inf"))
    rho = d_pp.kthvalue(k, dim=1).values.clamp_min(eps)
    d_pq = torch.cdist(p, q)
    nu = d_pq.kthvalue(k, dim=1).values.clamp_min(eps)
    kl = d * (nu.log() - rho.log()).mean() + math.log(q.shape[0] / (p.shape[0] - 1))
    return float(kl.item())
