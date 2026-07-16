"""Feynman-Kac guidance for prototype generative samplers."""

import math
from pathlib import Path
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_guidance_dataset(
    model,
    reward: Callable[[torch.Tensor], torch.Tensor],
    guidance_lambda: float,
    n_samples: int = 10_000,
    n_rollouts: int = 32,
    batch_size: int = 256,
    cache: str | Path | None = None,
) -> TensorDataset:
    """Create and optionally cache samples (t, x_t, log E[exp(lambda r(X_T)) | x_t])."""
    kind = _check_model(model)
    guidance_lambda = float(guidance_lambda)
    if not math.isfinite(guidance_lambda):
        raise ValueError("guidance_lambda must be finite.")
    if min(n_samples, n_rollouts, batch_size) <= 0:
        raise ValueError("n_samples, n_rollouts and batch_size must be positive.")

    cache = Path(cache) if cache is not None else None
    if cache is not None and cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        return TensorDataset(data["t"], data["x"], data["target"])

    rollouts = 1 if kind == "flow" else int(n_rollouts)
    times, states, targets = [], [], []
    model._net.eval()

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            x_t, time, state = _sample_states(model, size, kind)
            terminal = _continue(model, x_t.repeat_interleave(rollouts, dim=0), _repeat_state(state, rollouts), kind)
            values = guidance_lambda * reward(terminal).reshape(size, rollouts)
            target = torch.logsumexp(values, dim=1) - math.log(rollouts)

            times.append(time.cpu())
            states.append(x_t.cpu())
            targets.append(target.cpu())

    dataset = TensorDataset(torch.cat(times), torch.cat(states), torch.cat(targets))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        t, x, target = dataset.tensors
        torch.save({"t": t, "x": x, "target": target}, cache)

    return dataset


class Guidance:
    """Learn log-desirability and sample the corresponding tilted process."""

    def __init__(self, model, net: nn.Module):
        self.kind = _check_model(model)
        self.model = model
        self.net = net.to(device=model._device, dtype=model._fdtype)

        model._net.eval()
        for parameter in model._net.parameters():
            parameter.requires_grad_(False)

    def fit(
        self,
        dataset: TensorDataset,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 3e-4,
    ) -> list[float]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        history = []

        for _ in range(epochs):
            self.net.train()
            total = 0.0

            for t, x, target in loader:
                t = t.to(self.model._device, self.model._fdtype)
                x = x.to(self.model._device, self.model._fdtype)
                target = target.to(self.model._device, self.model._fdtype)

                prediction = self.net(x, t).reshape_as(target)
                loss = torch.nn.functional.mse_loss(prediction, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total += loss.item() * len(x)

            history.append(total / len(dataset))

        return history

    def sample(
        self,
        n_samples: int,
        sample_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model._net.eval()
        self.net.eval()
        if self.kind == "ddpm":
            return _sample_ddpm_guided(self.model, self.net, n_samples, sample_source)
        if self.kind == "dlpm":
            return _sample_dlpm_guided(self.model, self.net, n_samples)
        if self.kind == "flow":
            return _sample_flow_guided(self.model, self.net, n_samples, sample_source)
        raise RuntimeError(f"Unsupported guidance kind: {self.kind}.")


def _check_model(model) -> str:
    if hasattr(model, "_get_eps_hat"):
        if getattr(model, "_sampler", None) != "ddpm":
            raise ValueError("DDPM guidance requires sampler='ddpm'.")
        return "ddpm"
    if hasattr(model, "_g_Sigma_hat_Gamma") and hasattr(model, "_Sigma_1_t"):
        return "dlpm"
    if getattr(model, "_family", None) == "flow":
        if getattr(model, "_sampler", None) != "euler":
            raise ValueError("Flow guidance currently requires sampler='euler'.")
        if getattr(model, "_sample_timesteps", None) is None:
            raise ValueError("Flow guidance requires fixed sample timesteps.")
        return "flow"
    raise TypeError("Guidance supports DDPM, DLPM, and fixed-step flow models.")


def _repeat_state(state: dict[str, torch.Tensor], n_repeats: int) -> dict[str, torch.Tensor]:
    return {key: value.repeat_interleave(n_repeats, dim=0) for key, value in state.items()}


def _sample_states(model, n_samples: int, kind: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if kind == "ddpm":
        step = torch.randint(1, model._n_steps + 1, (n_samples,), device=model._device)
        x_t = _sample_ddpm_states(model, step)
        time = (step.to(model._fdtype) / model._n_steps).unsqueeze(1)
        return x_t, time, {"step": step}

    if kind == "dlpm":
        step = torch.randint(1, model._n_steps, (n_samples,), device=model._device)
        x_t = _sample_dlpm_states(model, step)
        time = (step.to(model._fdtype) / model._n_steps).unsqueeze(1)
        return x_t, time, {"step": step}

    index = torch.randint(0, model._sample_timesteps.numel() - 1, (n_samples,), device=model._device)
    x_t = _sample_flow_states(model, index)
    time = model._sample_timesteps.index_select(0, index).reshape(-1, 1)
    return x_t, time, {"index": index}


def _continue(model, x: torch.Tensor, state: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
    if kind == "ddpm":
        return _continue_ddpm(model, x, state["step"])
    if kind == "dlpm":
        return _continue_dlpm(model, x, state["step"])
    return _continue_flow(model, x, state["index"])


def _sample_ddpm_states(model, step: torch.Tensor) -> torch.Tensor:
    x_0 = model.sample(len(step))
    noise = model._sample_source(len(step))
    alpha_bar = model._alpha_bar.index_select(0, step - 1)
    alpha_bar = model._expand_batch_scalar(alpha_bar, x_0)
    return alpha_bar.sqrt() * x_0 + (1.0 - alpha_bar).sqrt() * noise


def _continue_ddpm(model, x: torch.Tensor, start_step: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    for step in range(int(start_step.max().item()), 0, -1):
        active = start_step >= step
        if not active.any():
            continue
        x_active = x[active]
        t = torch.full((len(x_active), 1), step / model._n_steps, device=x.device, dtype=x.dtype)
        eps = model._get_eps_hat(x_active, t, step - 1)
        mean, variance = _ddpm_posterior(model, x_active, eps, step)
        x[active] = mean if step == 1 else mean + variance.sqrt() * torch.randn_like(mean)
    return x


def _sample_ddpm_guided(model, net: nn.Module, n_samples: int, sample_source: torch.Tensor | None) -> torch.Tensor:
    n_samples, x = model._resolve_sample_source(n_samples, sample_source)
    for step in range(model._n_steps, 0, -1):
        t = torch.full((n_samples, 1), step / model._n_steps, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            eps = model._get_eps_hat(x, t, step - 1)
            mean, variance = _ddpm_posterior(model, x, eps, step)
        gradient = _guidance_gradient(net, x, t)
        mean = mean + variance * gradient
        x = mean if step == 1 else mean + variance.sqrt() * torch.randn_like(x)
        x = x.detach()
    return x


def _ddpm_posterior(model, x: torch.Tensor, eps: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = step - 1
    alpha = model._alphas[index]
    alpha_bar = model._alpha_bar[index]
    beta = model._betas[index]
    mean = (x - beta / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
    variance = model._sigma_max**2 * model._sqrt_post_var[index].square().to(device=x.device, dtype=x.dtype)
    return mean, variance


def _sample_dlpm_states(model, step: torch.Tensor) -> torch.Tensor:
    x_0 = model.sample(len(step))
    eps = model._sample_source_default(len(step))
    gamma_1_t = model._expand_batch_scalar(model._gamma_1_t.index_select(0, step), x_0)
    sigma_1_t = model._expand_batch_scalar(model._sigma_1_t.index_select(0, step), x_0)
    return gamma_1_t * x_0 + sigma_1_t * eps


def _dlpm_sigma_path(model, n_samples: int) -> torch.Tensor:
    A_path = torch.stack([model._draw_A(n_samples).squeeze(-1) for _ in range(model._n_steps)], dim=0)
    return model._Sigma_1_t(A_path)


def _continue_dlpm(model, x: torch.Tensor, start_step: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    Sigma_1_t = _dlpm_sigma_path(model, len(x))
    for step in range(int(start_step.max().item()), 0, -1):
        active = start_step >= step
        if not active.any():
            continue
        x[active] = _dlpm_step(model, x[active], Sigma_1_t, step, active)
    return x


def _sample_dlpm_guided(model, net: nn.Module, n_samples: int) -> torch.Tensor:
    n_samples = int(n_samples)
    Sigma_1_t = _dlpm_sigma_path(model, n_samples)
    eps = model._sample_source_default(n_samples)
    x = model._sigma_1_t[model._n_steps - 1] * eps
    active = torch.ones(n_samples, device=model._device, dtype=torch.bool)
    for step in range(model._n_steps - 1, 0, -1):
        t = torch.full((n_samples, 1), step / model._n_steps, device=model._device, dtype=model._fdtype)
        mean, variance = _dlpm_step_mean_variance(model, x, Sigma_1_t, step, active)
        gradient = _guidance_gradient(net, x, t)
        x = mean + model._expand_batch_scalar(variance, x) * gradient
        if step > 1:
            x = x + model._expand_batch_scalar(variance.sqrt(), x) * torch.randn_like(x)
        x = x.detach()
    return x


def _dlpm_step(model, x: torch.Tensor, Sigma_1_t: torch.Tensor, step: int, active: torch.Tensor) -> torch.Tensor:
    mean, variance = _dlpm_step_mean_variance(model, x, Sigma_1_t, step, active)
    if step > 1:
        mean = mean + model._expand_batch_scalar(variance.sqrt(), mean) * torch.randn_like(mean)
    return mean


def _dlpm_step_mean_variance(
    model,
    x: torch.Tensor,
    Sigma_1_t: torch.Tensor,
    step: int,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.full((len(x), 1), step / model._n_steps, device=model._device, dtype=model._fdtype)
    Sigma_hat, gamma_t, Gamma_t = model._g_Sigma_hat_Gamma(Sigma_1_t, step)
    Sigma_hat = Sigma_hat[active]
    Gamma_t = Gamma_t[active]
    eps_hat = model._net(x, t)
    mean = (x - model._expand_batch_scalar(Gamma_t, x) * model._sigma_1_t[step] * eps_hat) / gamma_t
    return mean, Sigma_hat


def _sample_flow_states(model, index: torch.Tensor) -> torch.Tensor:
    x = model._sample_source(len(index))
    for i in range(int(index.max().item())):
        active = index > i
        if not active.any():
            continue
        x[active] = _flow_step(model, x[active], i)
    return x


def _continue_flow(model, x: torch.Tensor, start_index: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    for i in range(int(start_index.min().item()), model._sample_timesteps.numel() - 1):
        active = start_index <= i
        if not active.any():
            continue
        x[active] = _flow_step(model, x[active], i)
    return x


def _sample_flow_guided(model, net: nn.Module, n_samples: int, sample_source: torch.Tensor | None) -> torch.Tensor:
    _, x = model._resolve_sample_source(n_samples, sample_source)
    for i in range(model._sample_timesteps.numel() - 1):
        t = _flow_time_batch(model, x, i)
        dt = model._sample_timesteps[i + 1] - model._sample_timesteps[i]
        with torch.no_grad():
            velocity = model._net(x, t)
        x = x + dt * (velocity + _guidance_gradient(net, x, t))
        x = x.detach()
    return x


def _flow_step(model, x: torch.Tensor, index: int) -> torch.Tensor:
    t = _flow_time_batch(model, x, index)
    dt = model._sample_timesteps[index + 1] - model._sample_timesteps[index]
    return x + dt * model._net(x, t)


def _flow_time_batch(model, x: torch.Tensor, index: int) -> torch.Tensor:
    return model._sample_timesteps[index].reshape(1, 1).expand(x.shape[0], 1).to(device=x.device, dtype=x.dtype)


def _guidance_gradient(net: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        value = net(x_grad, t).reshape(x.shape[0])
        gradient, = torch.autograd.grad(value.sum(), x_grad)
    return gradient
