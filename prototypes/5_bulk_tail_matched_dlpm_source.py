import logging
from pathlib import Path
import torch
from genkit.diffusion import DLPMEps
from genkit.training import train
from _constants import ALPHA, DIM, N_MMD, N_STEPS, N_TRIALS
from _utils import (
    add_test_vs_true_sample,
    evaluate_model,
    load_alpha_stable,
    make_net,
    make_train_kwargs,
    save_tail_figure,
    setup,
)

log_path = Path(__file__).with_name("logs") / f"{Path(__file__).stem}.log"
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


########################################################################################################################
# Additional classes

class BulkTailMatchedDLPM(DLPMEps):
    """DLPM-Eps with x0 sampled from the same bulk/tail region as x1."""

    def __init__(self, *args, threshold, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_threshold = torch.as_tensor(threshold, device=self._device, dtype=self._fdtype).detach()

    def _is_tail(self, x):
        threshold = self.tail_threshold.to(device=x.device, dtype=x.dtype)
        return x.reshape(x.shape[0], -1).norm(dim=1) >= threshold

    def _sample_source_matching(self, tail_mask):
        out = torch.empty(tail_mask.numel(), *self._sample_shape, device=self._device, dtype=self._fdtype)
        remaining = torch.arange(tail_mask.numel(), device=self._device)
        while remaining.numel() > 0:
            candidates = self._sample_source_default(remaining.numel())
            keep = self._is_tail(candidates) == tail_mask.index_select(0, remaining)
            out[remaining[keep]] = candidates[keep]
            remaining = remaining[~keep]
        return out

    def _loss(self, x, z=None, t=None):
        if z is not None:
            raise ValueError("BulkTailMatchedDLPM samples x0 internally and does not accept z.")

        x_1 = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x shape (N, *{self._sample_shape}), got {tuple(x.shape)}")
        self._n = x_1.size(0)

        if t is None:
            t = torch.randint(1, self._n_steps, (self._n,), device=self._device, dtype=self._idtype)
        else:
            t = self._check_t(t, self._n)

        t_e = self._expand(t.view(1, 1, self._n)).reshape(-1)
        t_norm = self._expand((t / self._n_steps).view(1, 1, self._n)).reshape(-1, 1)
        x_1_e = x_1.view(1, 1, self._n, *self._sample_shape).expand(
            self._n_trial_A,
            self._n_trial_G,
            self._n,
            *self._sample_shape,
        ).reshape(-1, *self._sample_shape)

        eps = self._sample_source_matching(self._is_tail(x_1_e))
        gamma_1_t = self._expand_batch_scalar(self._gamma_1_t.index_select(0, t_e), x_1_e)
        sigma_1_t = self._expand_batch_scalar(self._sigma_1_t.index_select(0, t_e), x_1_e)
        eps_hat = self._net(gamma_1_t * x_1_e + sigma_1_t * eps, t_norm)
        return self._loss_fn(eps_hat, eps, t)


########################################################################################################################
# Main

device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
tail_quantiles = [0.5, 0.8, 0.9]
train_kwargs = make_train_kwargs(device)
model_kwargs = dict(alpha=alpha, dim=dim, n_steps=N_STEPS, n_trial_A=1, n_trial_G=1, reduce_type="mean", device=device)
train_radius = x_train.flatten(1).norm(dim=1)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s tail_quantiles=%s", device, dtype, dim, n_trials, n_tail, n_mmd, tail_quantiles)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {
        "DLPM": DLPMEps(net=make_net(dim=dim, device=device, dtype=dtype), **model_kwargs),
    }
    for tail_quantile in tail_quantiles:
        threshold = torch.quantile(train_radius, tail_quantile).detach()
        models[f"Matched DLPM (q={tail_quantile:.1f})"] = BulkTailMatchedDLPM(
            net=make_net(dim=dim, device=device, dtype=dtype),
            threshold=threshold,
            **model_kwargs,
        )

    for name, model in models.items():
        logging.info("train/evaluate %s", name)
        model, _ = train(model, x_train, **train_kwargs)
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)


########################################################################################################################
# Plotting

save_tail_figure(rows, "Bulk/tail matched DLPM source", "5_bulk_tail_matched_dlpm_source.pdf")
logging.info("saved figure 5_bulk_tail_matched_dlpm_source.pdf")
