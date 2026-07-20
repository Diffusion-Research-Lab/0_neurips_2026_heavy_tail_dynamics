import logging
from pathlib import Path
import torch
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train
from _constants import DIM, FLOW_N_STEPS, FLOW_SAMPLE_STEPS, N_MMD, N_TRIALS
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


class BulkTailSplit:
    def __init__(self, x, tail_quantile=0.90):
        if x.ndim != 2:
            raise ValueError(f"Expected tabular samples with shape (N, dim), got {tuple(x.shape)}.")
        self.tail_quantile = float(tail_quantile)
        radius = x.flatten(1).norm(dim=1)
        self.threshold = torch.quantile(radius, self.tail_quantile)

        levels = self.assign(x)
        counts = torch.bincount(levels, minlength=2).to(dtype=x.dtype)
        self.level_probs = counts / counts.sum().clamp_min(1.0)

    def to(self, device=None, dtype=None):
        self.threshold = self.threshold.to(device=device, dtype=dtype)
        self.level_probs = self.level_probs.to(device=device, dtype=dtype)
        return self

    def assign(self, x):
        threshold = self.threshold.to(device=x.device, dtype=x.dtype)
        return (x.flatten(1).norm(dim=1) >= threshold).long()


class FixedHeteroscedasticGFL(GaussianFlowLinear):
    def __init__(self, *args, split, sigma_b=1.0, sigma_t=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split.to(device=self._device, dtype=self._fdtype)
        self.sigma = torch.tensor([sigma_b, sigma_t], device=self._device, dtype=self._fdtype)

    def _sample_source_at_level(self, level):
        eps = torch.randn(level.numel(), *self._sample_shape, device=self._device, dtype=self._fdtype)
        sigma = self.sigma.index_select(0, level)
        return self._expand_batch_scalar(sigma, eps) * eps

    def _precompute_loss(self, x, z=None, t=None):
        if z is not None:
            raise ValueError("FixedHeteroscedasticGFL samples x0 from its radial source and does not accept z.")
        x_1 = x.to(device=self._device, dtype=self._fdtype)

        # During training, the fixed source scale is tied to the target sample's bulk/tail region.
        x_0 = self._sample_source_at_level(self.split.assign(x_1))
        x_0, x_1, t = self._latent(x_1=x_1, x_0=x_0, t=t)
        t_data = self._expand_batch_scalar(t, x_1)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        return self._net(x_t, t), v_t, t

    def _sample_source_default(self, n_samples):
        level = torch.multinomial(self.split.level_probs, int(n_samples), replacement=True)
        return self._sample_source_at_level(level)


########################################################################################################################
# Main
device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
train_kwargs = make_train_kwargs(device)
flow_kwargs = dict(dim=dim, n_steps=FLOW_N_STEPS, t_min=0.0, t_max=1.0, sampler="euler", sample_steps=FLOW_SAMPLE_STEPS, device=device)

sigma_b = 1.0
sigma_t = 20.0
split = BulkTailSplit(x_train, tail_quantile=0.99)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s sigma_b=%s sigma_t=%s", device, dtype, dim, n_trials, n_tail, n_mmd, sigma_b, sigma_t)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {
        f"GFL (sigma={sigma_b:g})": GaussianFlowLinear(net=make_net(dim=dim, device=device, dtype=dtype), sigma_max=sigma_b, **flow_kwargs),
        f"GFL (sigma={sigma_t:g})": GaussianFlowLinear(net=make_net(dim=dim, device=device, dtype=dtype), sigma_max=sigma_t, **flow_kwargs),
        f"H-GFL (sigma_b={sigma_b:g} sigma_t={sigma_t:g})": FixedHeteroscedasticGFL(net=make_net(dim=dim, device=device, dtype=dtype), split=split, sigma_b=sigma_b, sigma_t=sigma_t, **flow_kwargs),
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
save_tail_figure(rows, "Heteroscedastic GF linear", figure_name)
logging.info("saved figure %s", figure_name)
