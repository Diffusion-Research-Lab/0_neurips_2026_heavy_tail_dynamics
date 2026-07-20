"""Feynman-Kac guidance for DDPM samplers."""

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
    """Sample (t, x_t) and regress log E[exp(lambda reward(X_0)) | x_t]."""
    _require_ddpm(model)

    guidance_lambda = float(guidance_lambda)
    n_samples = int(n_samples)
    n_rollouts = int(n_rollouts)
    batch_size = int(batch_size)

    if not math.isfinite(guidance_lambda):
        raise ValueError("guidance_lambda must be finite.")

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

            # Draw a noisy DDPM state x_t by first sampling x_0 from the current model.
            step = torch.randint(1, model._n_steps + 1, (size,), device=model._device)
            x_0 = model.sample(size)
            noise = model._sample_source(size)
            alpha_bar = model._expand_batch_scalar(model._alpha_bar.index_select(0, step - 1), x_0)
            x_t = alpha_bar.sqrt() * x_0 + (1.0 - alpha_bar).sqrt() * noise

            # Estimate E[exp(lambda reward(X_0)) | x_t] by continuing several reverse chains from the same x_t.
            terminal = x_t.repeat_interleave(n_rollouts, dim=0)
            start_step = step.repeat_interleave(n_rollouts)

            for reverse_step in range(int(start_step.max().item()), 0, -1):
                active = start_step >= reverse_step
                if not active.any():
                    continue

                x_active = terminal[active]
                t = torch.full(
                    (len(x_active), 1), reverse_step / model._n_steps, device=x_active.device, dtype=x_active.dtype,
                )
                eps = model._get_eps_hat(x_active, t, reverse_step - 1)
                mean, variance = _posterior(model, x_active, eps, reverse_step)
                terminal[active] = mean if reverse_step == 1 else mean + variance.sqrt() * torch.randn_like(mean)

            values = guidance_lambda * reward(terminal).reshape(size, n_rollouts)
            target = torch.logsumexp(values, dim=1) - math.log(n_rollouts)

            times.append((step.to(model._fdtype) / model._n_steps).unsqueeze(1).cpu())
            states.append(x_t.cpu())
            targets.append(target.cpu())

    dataset = TensorDataset(torch.cat(times), torch.cat(states), torch.cat(targets))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        t, x, target = dataset.tensors
        torch.save({"t": t, "x": x, "target": target}, cache)
    return dataset


class Guidance:
    """Learn a DDPM log-desirability function and sample the tilted reverse chain."""

    def __init__(self, model, net: nn.Module):
        _require_ddpm(model)

        self.model = model
        self.net = net.to(device=model._device, dtype=model._fdtype)

        model._net.eval()
        for parameter in model._net.parameters():
            parameter.requires_grad_(False)

    def fit(self, dataset: TensorDataset, epochs: int = 100, batch_size: int = 256, lr: float = 3e-4) -> list[float]:
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        history = []

        for _ in range(int(epochs)):
            self.net.train()
            total = 0.0

            for t, x, target in loader:
                t = t.to(self.model._device, self.model._fdtype)
                x = x.to(self.model._device, self.model._fdtype)
                target = target.to(self.model._device, self.model._fdtype)

                loss = torch.nn.functional.mse_loss(self.net(x, t).reshape_as(target), target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total += loss.item() * len(x)

            history.append(total / len(dataset))

        return history

    def sample(self, n_samples: int, sample_source: torch.Tensor | None = None) -> torch.Tensor:
        self.model._net.eval()
        self.net.eval()

        n_samples, x = self.model._resolve_sample_source(n_samples, sample_source)

        for step in range(self.model._n_steps, 0, -1):
            t = torch.full((n_samples, 1), step / self.model._n_steps, device=x.device, dtype=x.dtype)

            with torch.no_grad():
                eps = self.model._get_eps_hat(x, t, step - 1)
                mean, variance = _posterior(self.model, x, eps, step)

            # The Feynman-Kac tilt shifts the DDPM posterior mean by Var * grad log-desirability.
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                value = self.net(x_grad, t).reshape(n_samples)
                gradient, = torch.autograd.grad(value.sum(), x_grad)

            x = mean + variance * gradient
            if step > 1:
                x = x + variance.sqrt() * torch.randn_like(x)

            x = x.detach()

        return x


def _require_ddpm(model) -> None:
    if not hasattr(model, "_get_eps_hat"):
        raise TypeError("Guidance only supports DDPM models.")
    if getattr(model, "_sampler", None) != "ddpm":
        raise ValueError("Guidance requires a DDPM model with sampler='ddpm'.")


def _posterior(model, x: torch.Tensor, eps: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = model._alphas[step - 1]
    alpha_bar = model._alpha_bar[step - 1]
    beta = model._betas[step - 1]
    mean = (x - beta / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
    variance = model._sigma_max**2 * model._sqrt_post_var[step - 1].square().to(device=x.device, dtype=x.dtype)
    return mean, variance
