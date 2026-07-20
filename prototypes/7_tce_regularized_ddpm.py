import logging
from pathlib import Path
import torch
from genkit.diffusion import DDPMV
from genkit.training import train
from _constants import ALPHA, DIM, DTYPE, N_MMD, N_STEPS, N_TRIALS
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

DDPM_SIGMA_MAX = 5.0
REG_LAMBDA = 0.1
REG_TAIL_PROBS = torch.tensor([0.1, 0.01, 0.001], dtype=torch.float64)
REG_WEIGHTS = torch.ones_like(REG_TAIL_PROBS) / len(REG_TAIL_PROBS)
REG_DELTA = 1.0


########################################################################################################################
# Additional classes

class DDPMVWithTCEReg(DDPMV):
    """DDPM-v with a differentiable marginal upper-tail coverage penalty."""

    def __init__(self, *args, target_data, reg_tail_probs, reg_weights, reg_lambda, reg_delta, **kwargs):
        super().__init__(*args, **kwargs)

        reg_tail_probs = reg_tail_probs.to(device=self._device, dtype=self._fdtype)
        reg_weights = reg_weights.to(device=self._device, dtype=self._fdtype)
        if reg_tail_probs.ndim != 1:
            raise ValueError("reg_tail_probs must be one-dimensional.")
        if reg_weights.shape != reg_tail_probs.shape:
            raise ValueError("reg_weights must have the same shape as reg_tail_probs.")
        if torch.any((reg_tail_probs <= 0.0) | (reg_tail_probs >= 1.0)):
            raise ValueError("reg_tail_probs must satisfy 0 < p < 1.")
        if float(reg_delta) <= 0.0:
            raise ValueError("reg_delta must be positive.")

        target_flat = target_data.detach().to(device=self._device, dtype=self._fdtype).reshape(target_data.shape[0], -1)

        self._reg_lambda = float(reg_lambda)
        self._reg_delta = float(reg_delta)
        self._reg_tail_probs = reg_tail_probs
        self._reg_weights = reg_weights / reg_weights.sum().clamp_min(self._eps)
        self._reg_thresholds = torch.quantile(target_flat, 1.0 - self._reg_tail_probs, dim=0)

    def loss(self, x, z=None, t=None):
        x_1, x_t, eps, t_norm, t_idx, a_bar_t = self._latent(x_1=x, eps=z, t=t)
        v = torch.sqrt(a_bar_t) * eps - torch.sqrt(1.0 - a_bar_t) * x_1
        v_hat = self._net(x_t, t_norm)
        base_loss = self._reduce(self._loss_fn(v_hat, v, t_idx))

        x0_hat = torch.sqrt(a_bar_t) * x_t - torch.sqrt(1.0 - a_bar_t) * v_hat
        return base_loss + self._reg_lambda * self._tce_regularizer(x0_hat)

    def _tce_regularizer(self, x_gen):
        # Smooth the indicator 1{x > q_p} so the empirical upper-tail coverage is differentiable.
        x_flat = x_gen.reshape(x_gen.shape[0], -1)
        logits = (x_flat.unsqueeze(0) - self._reg_thresholds.unsqueeze(1)) / self._reg_delta
        coverage = torch.sigmoid(logits).mean(dim=1)
        target_coverage = self._reg_tail_probs.reshape(-1, 1)
        tce = (torch.log(coverage.clamp_min(self._eps)) - torch.log(target_coverage)).abs().mean(dim=1)
        return (self._reg_weights * tce).sum()


########################################################################################################################
# Main

device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
train_kwargs = make_train_kwargs(device)
ddpm_kwargs = dict(dim=dim, n_steps=N_STEPS, sigma_max=DDPM_SIGMA_MAX, sampler="ddpm", device=device, fdtype=DTYPE)
logging.info(
    "device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s lambda=%s reg_tail_probs=%s reg_delta=%s",
    device, dtype, dim, n_trials, n_tail, n_mmd, REG_LAMBDA, REG_TAIL_PROBS.tolist(), REG_DELTA,
)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {
        "DDPM": DDPMV(net=make_net(dim=dim, device=device, dtype=dtype), **ddpm_kwargs),
        f"DDPM+TCE reg (lambda={REG_LAMBDA:g})": DDPMVWithTCEReg(
            net=make_net(dim=dim, device=device, dtype=dtype),
            target_data=x_train,
            reg_tail_probs=REG_TAIL_PROBS,
            reg_weights=REG_WEIGHTS,
            reg_lambda=REG_LAMBDA,
            reg_delta=REG_DELTA,
            **ddpm_kwargs,
        ),
    }

    for name, model in models.items():
        logging.info("train/evaluate %s", name)
        model, _ = train(model, x_train, **train_kwargs)
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)


########################################################################################################################
# Plotting

figure_name = f"{Path(__file__).stem}.pdf"
save_tail_figure(rows, "DDPM TCE regularization", figure_name)
logging.info("saved figure %s", figure_name)
