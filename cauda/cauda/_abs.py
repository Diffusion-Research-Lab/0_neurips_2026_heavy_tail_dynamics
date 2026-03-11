"""Diffusion module."""

import torch
from ._sampling import sample_gaussian
from .utils import make_schedule


class Base:

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        base_or_sample: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = 'cpu',
    ):
        self._dim = int(dim)
        self._base_or_sample = base_or_sample
        self._n_steps = int(n_steps)

        self._dtype = dtype
        self._eps = torch.finfo(self._dtype).eps
        self._device = torch.device(device)
        self._net = net.to(device=self._device, dtype=self._dtype)

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError("'_sample_source_default' not implemented.")

    def _loss_fn(self, x_hat: torch.Tensor, x: torch.Tensor, t: int) -> torch.Tensor:
        raise NotImplementedError("'_loss_fn' not implemented.")

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("'_reduce' not implemented.")

    def _sample_source(self, n_samples: int) -> torch.Tensor:

        if isinstance(self._base_or_sample, torch.Tensor):
            base = self._base_or_sample.to(device=self._device, dtype=self._dtype)

            if base.ndim == 1:
                if base.numel() != self._dim:
                    raise ValueError(f"base_or_sample has dim {base.numel()}, expected {self._dim}")
                samples = base.unsqueeze(0).expand(n_samples, -1)
            else:
                idx = torch.randint(0, base.size(0), (n_samples,), device=self._device)
                samples = base.index_select(0, idx)

        else:
            samples = self._sample_source_default(n_samples)
        return samples


class DDPMAbstarct(Base):
    """DDPM abstract."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        base_or_sample: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = 'cpu',
    ):
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         dtype=dtype, device=device)

        self._betas, self._alphas, self._alpha_bar = make_schedule(n_steps, device, dtype)

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        return sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._dtype)

    def _latent(self, x_1: torch.Tensor, eps: torch.Tensor = None):
        x_1 = x_1.to(device=self._device, dtype=self._dtype)
        if x_1.size(-1) != self._dim:
            raise ValueError(f"Expected last dim {self._dim}, got {x_1.size(-1)}")

        n_samples = x_1.size(0)
        t = torch.randint(1, self._n_steps + 1, (n_samples,), device=self._device)
        t_norm = ((t - 1).to(self._dtype) / float(max(self._n_steps - 1, 1))).unsqueeze(-1)

        a_bar_t = self._alpha_bar.index_select(0, t).unsqueeze(-1)

        eps = self._sample_source(n_samples) if eps is None else eps.to(device=self._device,
                                                                        dtype=self._dtype)
        if eps.shape != x_1.shape:
            raise ValueError(f"eps must have shape {tuple(x_1.shape)}, got {tuple(eps.shape)}")

        x_t = a_bar_t.sqrt() * x_1 + (1.0 - a_bar_t).sqrt() * eps

        return x_1, x_t, eps, t_norm, t, a_bar_t

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        return loss_values.mean()


class FlowAbstract(Base):
    """Abstract source flow (Flow Matching)."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.01,
        t_max: float = 0.99,
        base_or_sample: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
    ):
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         dtype=dtype, device=device)

        self._t_min = float(t_min)
        self._t_max = float(t_max)
        if not (0.0 <= self._t_min < self._t_max <= 1.0):
            raise ValueError(f"Need 0 <= t_min < t_max <= 1, got {self._t_min}, {self._t_max}")

    def _t(self, n_samples):
        t = torch.rand((n_samples, 1), device=self._device, dtype=self._dtype)
        return self._t_min + (self._t_max - self._t_min) * t

    def _latent(self, x_1: torch.Tensor, x_0: torch.Tensor = None) -> torch.Tensor:
        x_1 = x_1.to(device=self._device, dtype=self._dtype)
        if x_1.ndim != 2 or x_1.size(1) != self._dim:
            raise ValueError(f"Expected x1 shape (N,{self._dim}), got {tuple(x_1.shape)}")

        n_samples = x_1.size(0)
        t = self._t(n_samples)

        x_0 = self._sample_source(n_samples) if x_0 is None else x_0.to(device=self._device,
                                                                        dtype=self._dtype)
        if x_0.shape != x_1.shape:
            raise ValueError(f"x0 must have shape {tuple(x_1.shape)}, got {tuple(x_0.shape)}")

        return x_0, x_1, t

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        self._net.eval()

        x = self._sample_source(n_samples)

        t0, t1 = self._t_min, self._t_max
        dt = (t1 - t0) / float(self._n_steps)
        t = torch.full((n_samples, 1), t0, device=self._device, dtype=self._dtype)

        for _ in range(self._n_steps):
            v0 = self._net(x, t)

            t_next = (t + dt).clamp_max(t1)
            x_euler = x + dt * v0

            v1 = self._net(x_euler, t_next)

            x = x + 0.5 * dt * (v0 + v1)
            t = t_next

        return x


class GaussianFlowAbstract(FlowAbstract):
    """Abstract Gaussian source flow (Flow Matching)."""

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        return sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._dtype)

    def _loss_fn(self, u_t: torch.Tensor, v_t: torch.Tensor, t: int) -> torch.Tensor:
        return torch.nn.functional.mse_loss(u_t, v_t, reduction='none')

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        return loss_values.mean()
