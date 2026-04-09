"""Post-hoc inspection utilities."""

from typing import Tuple
import numpy as np
import torch
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def model_est_err_curve(
    gen_model: object,
    x: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the model loss across its native time grid."""
    if not hasattr(gen_model, "_family"):
        raise ValueError(f"'inspect' utilities require a '._family' tag, got {type(gen_model)}.")
    x_target = x
    x_source = gen_model._sample_source(len(x_target))
    if gen_model._family == "flow":
        t_grid = torch.linspace(0.0, 1.0, steps=gen_model._n_steps, device=x.device, dtype=gen_model._fdtype)
    elif gen_model._family == "diffusion":
        t_grid = torch.arange(1, gen_model._n_steps + 1, device=x.device, dtype=gen_model._idtype)
    else:
        raise ValueError(f"'inspect' utilities can only inspect 'diffusion' or 'flow' models, got {gen_model._family}.")

    was_training = gen_model._net.training
    gen_model._net.eval()

    loss_values = []
    for t in t_grid:
        loss_values.append(gen_model._loss(x=x_target, z=x_source, t=t))

    if was_training:
        gen_model._net.train()

    return torch.stack(loss_values).cpu().numpy(), t_grid.cpu().numpy()


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
    if max_n_steps is not None and max_n_steps < 1:
        raise ValueError("max_n_steps must be at least 1 or None.")
    if not hasattr(gen_model, "_family"):
        raise ValueError(f"'inspect' utilities require a '._family' tag, got {type(gen_model)}.")

    x_eval = x.to(device=x.device, dtype=gen_model._fdtype)
    if gen_model._family == "flow":
        t_grid = torch.arange(gen_model._n_steps, device=x_eval.device, dtype=gen_model._idtype)
        time_scale = float(max(gen_model._n_steps - 1, 1))
    elif gen_model._family == "diffusion":
        t_grid = torch.arange(1, gen_model._n_steps + 1, device=x_eval.device, dtype=gen_model._idtype)
        time_scale = float(gen_model._n_steps)
    else:
        raise ValueError(f"'inspect' utilities can only inspect 'diffusion' or 'flow' models, got {gen_model._family}.")
    if max_n_steps is not None and max_n_steps < len(t_grid):
        idx = torch.linspace(0, len(t_grid) - 1, steps=int(max_n_steps), device=t_grid.device)
        t_grid = t_grid.index_select(0, torch.round(idx).to(dtype=torch.long))
    was_training = gen_model._net.training
    gen_model._net.eval()

    try:
        curve = []
        for t_step in t_grid:
            t_batch = (t_step.to(dtype=gen_model._fdtype) / time_scale).reshape(1, 1)
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
            curve.append(torch.stack(sample_values).max())
    finally:
        if was_training:
            gen_model._net.train()

    return torch.stack(curve).cpu().numpy(), t_grid.cpu().numpy()


def fit_hmm_on_weight_stats(weight_stats, n_states=None, random_state=0):
    """Fit an HMM on per-epoch weight statistics to segment training phases."""
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
