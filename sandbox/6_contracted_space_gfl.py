import time

import torch
from _utils import (
    add_test_vs_true_sample,
    evaluate_model,
    evaluation_sizes,
    load_alpha_stable,
    make_mlp,
    make_train_kwargs,
    save_tail_figure,
    setup,
)
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train


########################################################################################################################
# Setup

device, dtype = setup()


########################################################################################################################
# Additional classes

class AsinhContraction:
    def __init__(self, scale, inverse_clip):
        self.scale = scale
        self.inverse_clip = float(inverse_clip)
        self.inverse_amplification_reference = torch.ones((), device=scale.device, dtype=scale.dtype)

    @classmethod
    def fit(cls, x, scale_quantile=0.5, inverse_clip_quantile=0.9999, inverse_clip_margin=0.75):
        x_flat = x.reshape(x.shape[0], -1)
        eps = torch.finfo(x.dtype).eps
        scale = torch.quantile(x_flat.abs(), float(scale_quantile), dim=0).clamp_min(eps)
        y_flat = torch.asinh(x_flat / scale)
        inverse_clip = torch.quantile(y_flat.abs().reshape(-1), float(inverse_clip_quantile))
        inverse_clip = float(inverse_clip.item() + float(inverse_clip_margin))
        return cls(scale=scale.detach(), inverse_clip=inverse_clip)

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

    def fit_inverse_amplification_reference(self, y):
        amp = self.inverse_amplification(y)
        self.inverse_amplification_reference = torch.median(amp).detach().clamp_min(torch.finfo(y.dtype).eps)
        return self

    def summary(self):
        return {
            "scale_min": float(self.scale.min().item()),
            "scale_max": float(self.scale.max().item()),
            "inverse_clip": self.inverse_clip,
            "inverse_amp_ref": float(self.inverse_amplification_reference.item()),
            "inverse_abs_max": float((self.scale.max() * torch.sinh(torch.tensor(self.inverse_clip, device=self.scale.device, dtype=self.scale.dtype))).item()),
        }


class ContractedSpaceGFL:
    def __init__(self, flow, contraction):
        self.flow = flow
        self.contraction = contraction

    def __getattr__(self, name):
        if "flow" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.flow, name)

    @torch.no_grad()
    def sample(self, n_samples=None, sample_source=None, chunk_size=None):
        y = self.flow.sample(n_samples=n_samples, sample_source=sample_source, chunk_size=chunk_size)
        return self.contraction.inverse(y)


class InverseJacobianWeightedGFL(GaussianFlowLinear):
    def __init__(self, *args, contraction, weight_power=0.5, weight_max=8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.contraction = contraction
        self.weight_power = float(weight_power)
        self.weight_max = float(weight_max)
        self.amp_ref = self.contraction.inverse_amplification_reference.to(device=self._device, dtype=self._fdtype)

    def _loss(self, x, z=None, t=None):
        v_t_hat, v_t, _ = self._precompute_loss(x=x, z=z, t=t)
        y_1 = x.to(device=self._device, dtype=self._fdtype).reshape(x.shape[0], -1)
        amp = self.contraction.inverse_amplification(y_1).to(device=self._device, dtype=self._fdtype)
        weight = (amp / self.amp_ref.clamp_min(self._eps)).clamp_min(1.0).pow(self.weight_power).clamp_max(self.weight_max)
        return weight.view(-1, *([1] * (x.ndim - 1))) * torch.nn.functional.mse_loss(v_t_hat, v_t, reduction="none")


def build_gfl(sigma_max=1.0, sample_steps=128):
    return GaussianFlowLinear(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=128,
        t_min=0.0,
        t_max=1.0,
        sigma_max=float(sigma_max),
        sampler="euler",
        sample_steps=int(sample_steps),
        device=device,
    )


def build_weighted_contracted_gfl(contraction, sigma_max=1.0, sample_steps=512, weight_power=0.5, weight_max=8.0):
    return InverseJacobianWeightedGFL(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=128,
        t_min=0.0,
        t_max=1.0,
        sigma_max=float(sigma_max),
        sampler="euler",
        sample_steps=int(sample_steps),
        device=device,
        contraction=contraction,
        weight_power=weight_power,
        weight_max=weight_max,
    )


########################################################################################################################
# Main

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_tail, n_mmd = evaluation_sizes(x_test)
n_trials = 10
train_kwargs = make_train_kwargs(device)

contraction = AsinhContraction.fit(
    x_train,
    scale_quantile=0.5,
    inverse_clip_quantile=0.9999,
    inverse_clip_margin=0.75,
)
y_train = contraction.transform(x_train)
contraction.fit_inverse_amplification_reference(y_train)
y_sigma = float(y_train.flatten(1).square().mean().sqrt().clamp_min(0.05).item())

print(f"[INFO] n_trials={n_trials} train_kwargs={train_kwargs}")
print(f"[INFO] contraction={contraction.summary()} y_sigma={y_sigma:.4g} weight_power=0.5 weight_max=8")

rows = []
for trial in range(1, n_trials + 1):
    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)

    models = {
        "GF linear": build_gfl(sigma_max=1.0, sample_steps=128),
        "GF linear + asinh contract": ContractedSpaceGFL(
            build_weighted_contracted_gfl(contraction, sigma_max=y_sigma, sample_steps=512, weight_power=0.5, weight_max=8.0),
            contraction,
        ),
    }

    print(f"[INFO] trial {trial}/{n_trials}: train GF linear")
    models["GF linear"], _ = train(models["GF linear"], x_train, **train_kwargs)

    start_time = time.time()
    print(f"[INFO] trial {trial}/{n_trials}: train GF linear + asinh contract")
    models["GF linear + asinh contract"].flow, _ = train(models["GF linear + asinh contract"].flow, y_train, **train_kwargs)
    print(f"[INFO] trial {trial}/{n_trials}: trained contracted model in {time.time() - start_time:.1f} s")

    for name, model in models.items():
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    print(f"[INFO] trial {trial}/{n_trials}: done")


########################################################################################################################
# Plotting

add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)
save_tail_figure(rows, "Contracted-space GF linear", "6_contracted_space_gfl.pdf")
