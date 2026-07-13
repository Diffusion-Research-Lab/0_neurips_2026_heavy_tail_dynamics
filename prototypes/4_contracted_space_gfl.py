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

class AsinhContraction:
    def __init__(self, scale, inverse_clip, amp_ref):
        self.scale = scale.detach()
        self.inverse_clip = float(inverse_clip)
        self.amp_ref = amp_ref.detach()

    @classmethod
    def fit(cls, x, scale_quantile=0.5, inverse_clip_quantile=0.9999, inverse_clip_margin=0.75):
        x_flat = x.reshape(x.shape[0], -1)
        eps = torch.finfo(x.dtype).eps
        scale = torch.quantile(x_flat.abs(), float(scale_quantile), dim=0).clamp_min(eps)
        y_flat = torch.asinh(x_flat / scale)
        inverse_clip = torch.quantile(y_flat.abs().reshape(-1), float(inverse_clip_quantile))
        inverse_clip = float(inverse_clip.item() + float(inverse_clip_margin))
        amp = scale * torch.cosh(y_flat.clamp(-inverse_clip, inverse_clip))
        amp_ref = amp.square().mean(dim=1).sqrt().median().clamp_min(eps)
        return cls(scale=scale, inverse_clip=inverse_clip, amp_ref=amp_ref)

    def transform(self, x):
        x_flat = x.reshape(x.shape[0], -1)
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        return torch.asinh(x_flat / scale).reshape_as(x)

    def inverse(self, y):
        y_flat = y.reshape(y.shape[0], -1)
        scale = self.scale.to(device=y.device, dtype=y.dtype)
        y_flat = y_flat.clamp(-self.inverse_clip, self.inverse_clip)
        return (scale * torch.sinh(y_flat)).reshape_as(y)

    def inverse_amplification(self, y):
        y_flat = y.reshape(y.shape[0], -1)
        scale = self.scale.to(device=y.device, dtype=y.dtype)
        amp = scale * torch.cosh(y_flat.clamp(-self.inverse_clip, self.inverse_clip))
        return amp.square().mean(dim=1).sqrt()


class ContractedWGFL(GaussianFlowLinear):
    def __init__(self, *args, contraction, **kwargs):
        super().__init__(*args, **kwargs)
        self.contraction = contraction
        self.amp_ref = contraction.amp_ref.to(device=self._device, dtype=self._fdtype)

    def _loss(self, x, z=None, t=None):
        v_t_hat, v_t, _ = self._precompute_loss(x=x, z=z, t=t)
        y = x.to(device=self._device, dtype=self._fdtype)
        amp = self.contraction.inverse_amplification(y)
        weight = (amp / self.amp_ref.clamp_min(self._eps)).clamp_min(1.0).sqrt().clamp_max(8.0)
        loss = torch.nn.functional.mse_loss(v_t_hat, v_t, reduction="none")
        return self._expand_batch_scalar(weight, loss) * loss

    @torch.no_grad()
    def sample(self, n_samples=None, sample_source=None, chunk_size=None):
        y = super().sample(n_samples=n_samples, sample_source=sample_source, chunk_size=chunk_size)
        return self.contraction.inverse(y)


########################################################################################################################
# Main
device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
train_kwargs = make_train_kwargs(device)
flow_kwargs = dict(dim=dim, n_steps=FLOW_N_STEPS, t_min=0.0, t_max=1.0, sampler="euler", device=device)

contraction = AsinhContraction.fit(x_train, scale_quantile=0.5, inverse_clip_quantile=0.9999, inverse_clip_margin=0.75)
y_train = contraction.transform(x_train)
y_sigma = float(y_train.flatten(1).square().mean().sqrt().clamp_min(0.05).item())
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s y_sigma=%s", device, dtype, dim, n_trials, n_tail, n_mmd, y_sigma)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {
        "GFL": GaussianFlowLinear(net=make_net(dim=dim, device=device, dtype=dtype), sigma_max=1.0, sample_steps=FLOW_SAMPLE_STEPS, **flow_kwargs),
        "H+GFL": ContractedWGFL(net=make_net(dim=dim, device=device, dtype=dtype), sigma_max=y_sigma, sample_steps=512, contraction=contraction, **flow_kwargs),
    }

    for ((name, model), z_train) in zip(models.items(), [x_train, y_train]):
        logging.info("train/evaluate %s", name)
        model, _ = train(model, z_train, **train_kwargs)
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)


########################################################################################################################
# Plotting

save_tail_figure(rows, "Contracted-space GF linear", "4_contracted_space_gfl.pdf")
logging.info("saved figure 4_contracted_space_gfl.pdf")
