import math
import numpy as np
import torch
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train
from _utils import (
    add_test_vs_true_sample,
    evaluation_sizes,
    evaluate_model_with_source,
    load_alpha_stable,
    make_mlp,
    make_train_kwargs,
    save_tail_figure,
    setup,
)


########################################################################################################################
# Setup

device, dtype = setup()


########################################################################################################################
# Additional classes and helpers

def _time_batch(t, x):
    if not isinstance(t, torch.Tensor):
        return torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
    return t.to(device=x.device, dtype=x.dtype).reshape(1, 1).expand(x.size(0), 1)


def _check_timesteps(timesteps, x):
    timesteps = torch.as_tensor(timesteps, device=x.device, dtype=x.dtype)
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("timesteps must be a 1D tensor with at least two entries.")
    return timesteps


class PathSMC:
    def __init__(self, n_particles, beta_max=0.005, ess_frac=0.5, device="cpu", dtype=torch.float32):
        self.beta_max = float(beta_max)
        self.ess_frac = float(ess_frac)
        self.logw = torch.full((n_particles,), -math.log(n_particles), device=device, dtype=dtype)
        self.logh_prev = torch.zeros(n_particles, device=device, dtype=dtype)
        self.logZ = torch.zeros((), device=device, dtype=dtype)
        self.ess_history = []
        self.resampling_steps = []
        self.n_resampled = 0

    def T(self, x):
        return x.flatten(1).norm(dim=1)

    def weights(self):
        return self.logw.exp()

    def ess(self):
        w = self.weights()
        return 1.0 / w.square().sum()

    def systematic_resample_idx(self):
        n = self.logw.numel()
        positions = (torch.rand((), device=self.logw.device, dtype=self.logw.dtype) + torch.arange(n, device=self.logw.device, dtype=self.logw.dtype)) / n
        cdf = self.weights().cumsum(0)
        cdf[-1] = 1.0
        return torch.searchsorted(cdf, positions).clamp(max=n - 1)

    def resample(self, x, step):
        idx = self.systematic_resample_idx()
        self.logw.fill_(-math.log(self.logw.numel()))
        self.logh_prev = self.logh_prev.index_select(0, idx)
        self.resampling_steps.append(int(step))
        self.n_resampled += int(x.size(0))
        return x.index_select(0, idx)

    def update(self, x, proxy_x, step, n_steps):
        beta_k = self.beta_max * step / n_steps
        logh = beta_k * self.T(proxy_x)
        logG = logh - self.logh_prev

        logZ_inc = torch.logsumexp(self.logw + logG, dim=0)
        self.logZ = self.logZ + logZ_inc
        self.logw = self.logw + logG - logZ_inc
        self.logh_prev = logh

        ess = self.ess()
        self.ess_history.append(float(ess))
        if ess < self.ess_frac * x.size(0):
            x = self.resample(x, step)
        return x


@torch.inference_mode()
def sample_flow_euler_path_smc(net, x, timesteps, smc, return_trajectory=True, final_resample=True):
    timesteps = _check_timesteps(timesteps, x)
    trajectory = [x] if return_trajectory else None
    n_steps = timesteps.numel() - 1
    for i in range(n_steps):
        t = _time_batch(timesteps[i], x)
        dt = timesteps[i + 1] - timesteps[i]
        x = x + dt * net(x, t)
        x = smc.update(x, proxy_x=x, step=i + 1, n_steps=n_steps)
        if trajectory is not None:
            trajectory.append(x)

    if final_resample:
        x = smc.resample(x, n_steps)
        if trajectory is not None:
            trajectory[-1] = x
    return x, trajectory


class GaussianFlowLinearPathSMC(GaussianFlowLinear):
    def __init__(self, base, beta_max=0.005, ess_frac=0.5, final_resample=True):
        super().__init__(
            base._net,
            dim=base._dim,
            n_steps=base._n_steps,
            t_min=base._t_min,
            t_max=base._t_max,
            sigma_max=base._sigma_max,
            sampler=base._sampler,
            schedule=base._schedule,
            sample_steps=base._sample_steps,
            device=base._device,
            fdtype=base._fdtype,
            idtype=base._idtype,
        )
        self.beta_max = float(beta_max)
        self.ess_frac = float(ess_frac)
        self.final_resample = bool(final_resample)
        self.last_smc = None

    @torch.inference_mode()
    def _sample(self, n_samples=None, return_trajectory=True, sample_source=None, chunk_size=None):
        if self._sampler != "euler":
            raise ValueError("Path SMC sampler is implemented only for Euler flow sampling.")
        self._net.eval()
        _, x = self._resolve_sample_source(n_samples, sample_source)
        smc = PathSMC(x.size(0), beta_max=self.beta_max, ess_frac=self.ess_frac, device=x.device, dtype=x.dtype)
        x, trajectory = sample_flow_euler_path_smc(
            self._net,
            x,
            self._sample_timesteps,
            smc,
            return_trajectory=return_trajectory,
            final_resample=self.final_resample,
        )
        self.last_smc = smc
        return x, trajectory

    @torch.inference_mode()
    def sample(self, n_samples=None, sample_source=None, chunk_size=None):
        x, _ = self._sample(n_samples, return_trajectory=False, sample_source=sample_source, chunk_size=chunk_size)
        return x

    @torch.inference_mode()
    def sample_smc(self, n_samples=None, sample_source=None, chunk_size=None):
        x, _ = self._sample(n_samples, return_trajectory=False, sample_source=sample_source, chunk_size=chunk_size)
        return x, self.last_smc


def build_base_model():
    return GaussianFlowLinear(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=128,
        t_min=0.0,
        t_max=1.0,
        sigma_max=1.0,
        sampler="euler",
        sample_steps=128,
        device=device,
    )


########################################################################################################################
# Main

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_tail, n_mmd = evaluation_sizes(x_test)
n_trials = 10
sample_chunk_size = 10_000
smc_beta_max = 0.005
smc_ess_frac = 0.5
train_kwargs = make_train_kwargs(device)

print(f"[INFO] n_trials={n_trials} train_kwargs={train_kwargs}")
print(f"[INFO] smc_beta_max={smc_beta_max} smc_ess_frac={smc_ess_frac} sample_chunk_size={sample_chunk_size}")

rows = []
for trial in range(1, n_trials + 1):
    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")

    base = build_base_model()
    print(f"[INFO] trial {trial}/{n_trials}: train GF linear")
    base, _ = train(base, x_train, **train_kwargs)
    models = {
        "GF linear": base,
        "GF linear + path SMC": GaussianFlowLinearPathSMC(base, beta_max=smc_beta_max, ess_frac=smc_ess_frac, final_resample=True),
    }

    x_source = base._sample_source(n_tail)
    for name, model in models.items():
        chunk_size = None if isinstance(model, GaussianFlowLinearPathSMC) else sample_chunk_size
        row = evaluate_model_with_source(name, model, trial, x_test, x_source, n_tail, n_mmd, chunk_size=chunk_size)
        if isinstance(model, GaussianFlowLinearPathSMC):
            smc = model.last_smc
            assert torch.isfinite(smc.logZ)
            assert abs(smc.weights().sum().item() - 1.0) < 1e-4
            assert 1.0 <= smc.ess().item() <= n_tail * (1.0 + 1e-6)
            row["beta_max"] = model.beta_max
            row["ess_min"] = float(np.min(smc.ess_history))
            row["n_resampled"] = smc.n_resampled
            row["logZ"] = float(smc.logZ)
            print(
                f"[INFO] trial {trial}/{n_trials}: SMC logZ={row['logZ']:.4g} "
                f"ess_min={row['ess_min']:.1f} n_resampled={row['n_resampled']}"
            )
        rows.append(row)

    print(f"[INFO] trial {trial}/{n_trials}: done")


########################################################################################################################
# Plotting

add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)
save_tail_figure(rows, "Path-space SMC tail correction", "5_smc_tail_correction_model.pdf")
