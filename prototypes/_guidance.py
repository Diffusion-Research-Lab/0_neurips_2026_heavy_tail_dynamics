"""Feynman-Kac guidance for stochastic DDPM sampling."""

import math
from pathlib import Path
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_guidance_dataset(
    model,
    reward: Callable[[torch.Tensor], torch.Tensor],
    temperature: float = 1.0,
    n_samples: int = 10_000,
    n_rollouts: int = 32,
    batch_size: int = 256,
    cache: str | Path | None = None,
) -> TensorDataset:
    """Create and optionally cache samples (t, x_t, log h(t, x_t))."""
    _check_model(model)

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if min(n_samples, n_rollouts, batch_size) <= 0:
        raise ValueError("n_samples, n_rollouts and batch_size must be positive.")

    cache = Path(cache) if cache is not None else None
    if cache is not None and cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        return TensorDataset(data["t"], data["x"], data["target"])

    times, states, targets = [], [], []
    model._net.eval()

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            step = torch.randint(1, model._n_steps + 1, (size,), device=model._device)
            x_t = _sample_states(model, step)

            terminal = _continue(
                model,
                x_t.repeat_interleave(n_rollouts, dim=0),
                step.repeat_interleave(n_rollouts),
            )

            values = reward(terminal).reshape(size, n_rollouts) / temperature
            target = torch.logsumexp(values, dim=1) - math.log(n_rollouts)

            times.append((step / model._n_steps).unsqueeze(1).cpu())
            states.append(x_t.cpu())
            targets.append(target.cpu())

    dataset = TensorDataset(torch.cat(times), torch.cat(states), torch.cat(targets))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        t, x, target = dataset.tensors
        torch.save({"t": t, "x": x, "target": target}, cache)

    return dataset


class Guidance:
    """Learn log-desirability and use its gradient to guide DDPM sampling."""

    def __init__(self, model, net: nn.Module):
        _check_model(model)
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
        strength: float = 1.0,
        sample_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model._net.eval()
        self.net.eval()

        n_samples, x = self.model._resolve_sample_source(n_samples, sample_source)

        for step in range(self.model._n_steps, 0, -1):
            t = torch.full((n_samples, 1), step / self.model._n_steps, device=x.device, dtype=x.dtype)

            with torch.no_grad():
                eps = self.model._get_eps_hat(x, t, step - 1)
                mean, variance = _posterior(self.model, x, eps, step)

            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                value = self.net(x_grad, t).reshape(n_samples)
                gradient, = torch.autograd.grad(value.sum(), x_grad)

            mean = mean + strength * variance * gradient
            x = mean if step == 1 else mean + variance.sqrt() * torch.randn_like(x)
            x = x.detach()

        return x


def _check_model(model):
    if not hasattr(model, "_get_eps_hat"):
        raise TypeError("Guidance only supports DDPM models.")
    if getattr(model, "_sampler", None) != "ddpm":
        raise ValueError("Guidance requires sampler='ddpm'.")


def _sample_states(model, step):
    """Sample x_t by generating x_0 and applying the forward DDPM kernel."""
    x_0 = model.sample(len(step))
    noise = model._sample_source(len(step))
    alpha_bar = model._alpha_bar.index_select(0, step - 1)
    alpha_bar = model._expand_batch_scalar(alpha_bar, x_0)
    return alpha_bar.sqrt() * x_0 + (1.0 - alpha_bar).sqrt() * noise


def _continue(model, x, start_step):
    """Continue each state from its DDPM timestep to the final sample."""
    x = x.clone()

    for step in range(int(start_step.max()), 0, -1):
        active = start_step >= step
        if not active.any():
            continue

        x_active = x[active]
        t = torch.full((len(x_active), 1), step / model._n_steps, device=x.device, dtype=x.dtype)

        eps = model._get_eps_hat(x_active, t, step - 1)
        mean, variance = _posterior(model, x_active, eps, step)
        x[active] = mean if step == 1 else mean + variance.sqrt() * torch.randn_like(mean)

    return x


def _posterior(model, x, eps, step):
    index = step - 1
    alpha = model._alphas[index]
    alpha_bar = model._alpha_bar[index]
    beta = model._betas[index]

    mean = (x - beta / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
    variance = model._sigma_max**2 * model._sqrt_post_var[index].square().to(device=x.device, dtype=x.dtype)
    return mean, variance
