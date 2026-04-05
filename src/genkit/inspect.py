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
