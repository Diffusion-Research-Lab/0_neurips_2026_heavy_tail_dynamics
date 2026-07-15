import logging
from pathlib import Path
import torch
from genkit._solvers import sample_flow
from genkit.diffusion import DLPMEps
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train
from _constants import ALPHA, DIM, FLOW_N_STEPS, FLOW_SAMPLE_STEPS, N_MMD, N_STEPS, N_TRIALS
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

class TailFlagGFL(GaussianFlowLinear):
    """GFL with a binary endpoint tail flag appended to the vector-field input."""

    def __init__(self, *args, tail_threshold, tail_prob, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_threshold = torch.as_tensor(tail_threshold, device=self._device, dtype=self._fdtype).detach()
        self.tail_prob = float(tail_prob)

    def _tail_flag(self, x):
        radius = x.reshape(x.shape[0], -1).norm(dim=1, keepdim=True)
        threshold = self.tail_threshold.to(device=x.device, dtype=x.dtype)
        return (radius >= threshold).to(dtype=x.dtype)

    def _append_flag(self, x, flag):
        flag = flag.to(device=x.device, dtype=x.dtype).reshape(x.shape[0], 1)
        return torch.cat([x.reshape(x.shape[0], -1), flag], dim=1)

    def _loss(self, x, z=None, t=None):
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)
        t_data = self._expand_batch_scalar(t, x_0)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(self._append_flag(x_t, self._tail_flag(x_1)), t).reshape_as(v_t)
        return self._loss_fn(v_t_hat, v_t, t)

    @torch.inference_mode()
    def _sample(self, n_samples=None, return_trajectory=True, sample_source=None, chunk_size=None):
        if chunk_size is not None:
            raise ValueError("TailFlagGFL does not support chunked sampling.")

        self._net.eval()
        n_samples, x = self._resolve_sample_source(n_samples, sample_source)
        flag = (torch.rand(n_samples, 1, device=self._device, dtype=self._fdtype) < self.tail_prob).to(self._fdtype)

        def field(x_t, t):
            return self._net(self._append_flag(x_t, flag), t).reshape_as(x_t)

        if self._sampler == "adaptive_heun":
            return sample_flow(
                field,
                x,
                sampler=self._sampler,
                t_min=self._t_min,
                t_max=self._t_max,
                atol=self._atol,
                rtol=self._rtol,
                h_init=self._h_init,
                h_min=self._h_min,
                h_max=self._h_max,
                return_trajectory=return_trajectory,
            )
        return sample_flow(field, x, self._sample_timesteps, sampler=self._sampler, return_trajectory=return_trajectory)


########################################################################################################################
# Main

device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
tail_quantile = 0.9
train_kwargs = make_train_kwargs(device)
train_radius = x_train.flatten(1).norm(dim=1)
tail_threshold = torch.quantile(train_radius, tail_quantile).detach()
tail_prob = (train_radius >= tail_threshold).to(dtype=dtype).mean().item()
dlpm_kwargs = dict(alpha=alpha, dim=dim, n_steps=N_STEPS, n_trial_A=1, n_trial_G=1, reduce_type="mean", device=device)
flow_kwargs = dict(dim=dim, n_steps=FLOW_N_STEPS, t_min=0.0, t_max=1.0, sigma_max=1.0, sampler="euler", sample_steps=FLOW_SAMPLE_STEPS, device=device)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s tail_quantile=%s tail_prob=%s", device, dtype, dim, n_trials, n_tail, n_mmd, tail_quantile, tail_prob)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {
        "DLPM": DLPMEps(net=make_net(dim=dim, device=device, dtype=dtype), **dlpm_kwargs),
        "GFL": GaussianFlowLinear(net=make_net(dim=dim, device=device, dtype=dtype), **flow_kwargs),
        f"Tail-flag GFL (q={tail_quantile:.1f})": TailFlagGFL(
            net=make_net(dim=dim, input_dim=dim + 1, output_dim=dim, device=device, dtype=dtype),
            tail_threshold=tail_threshold,
            tail_prob=tail_prob,
            **flow_kwargs,
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

save_tail_figure(rows, "Tail-conditioned GFL", "6_tail_conditioned_gfl.pdf")
logging.info("saved figure 6_tail_conditioned_gfl.pdf")
