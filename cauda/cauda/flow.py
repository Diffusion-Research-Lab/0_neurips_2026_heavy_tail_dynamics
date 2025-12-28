"""Flow module."""

# Authors: Hamza Cherkaoui

from typing import Tuple
import math
import torch
from ._abs import FlowAbstract, GaussianFlowAbstract
from ._sampling import sample_scaled_scalar_alpha_stable


class AlphaStableFlowLinear(FlowAbstract):
    """Alpha source flow with a linear path (Flow Matching)."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.01,
        t_max: float = 0.99,
        alpha: float = 1.8,
        reduce_type: str = "mean",
        clamp_A: Tuple[float, float] = (0.0, 1e3),
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

        if reduce_type not in ("mean", "median"):
            raise ValueError(f"'reduce_type' must be in ('mean','median'), got {reduce_type}.")
        self._reduce_type = reduce_type

        self._a = float(alpha)
        if not (0.0 < self._a < 2.0):
            raise ValueError(f"'alpha' must be in (0,2), got {self._a}.")

        self._clamp_A = clamp_A
        if (self._clamp_A is not None):
            if not (isinstance(self._clamp_A, (tuple, list)) and len(self._clamp_A) == 2):
                raise ValueError("clamp_A must be a (min,max) tuple or None.")
            self._A_min, self._A_max = float(self._clamp_A[0]), float(self._clamp_A[1])

    def _safe_A(self, A):
        A = A.clamp_min(self._eps)  # by default
        if self._clamp_A is not None:
            return A.clamp(min=self._A_min, max=self._A_max)
        else:
            return A

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        A = sample_scaled_scalar_alpha_stable(n_samples=n_samples, alpha=self._a,
                                              device=self._device, dtype=self._dtype)
        A = self._safe_A(A)
        G = torch.randn(n_samples, self._dim, device=self._device, dtype=self._dtype)
        return A.sqrt() * G

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        if self._reduce_type == "mean":
            return loss_values.sqrt().mean()
        return loss_values.sqrt().median()

    def _loss_fn(self, x_0_hat: torch.Tensor, x_0: torch.Tensor, t: int) -> torch.Tensor:
        return (x_0_hat - x_0).square().sum(dim=-1)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        x_1 = x.to(device=self._device, dtype=self._dtype)
        if x_1.ndim != 2 or x_1.size(1) != self._dim:
            raise ValueError(f"Expected x1 shape (N,{self._dim}), got {tuple(x_1.shape)}")
        n_samples = x_1.size(0)

        x_0 = self._sample_source(n_samples) if z is None else z.to(device=self._device,
                                                                    dtype=self._dtype)
        if x_0.shape != x_1.shape:
            raise ValueError(f"x0 must have shape {tuple(x_1.shape)}, got {tuple(x_0.shape)}")

        t = torch.rand((n_samples, 1), device=self._device, dtype=self._dtype)
        t = self._t_min + (self._t_max - self._t_min) * t

        x_t = (1.0 - t) * x_0 + t * x_1
        v_t = x_1 - x_0

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        return self._reduce(self._loss_fn(self._net(x_t, t), v_t, t))


class GaussianFlowLinear(GaussianFlowAbstract):
    """Gaussian source flow with a linear path (Flow Matching)."""

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        x_0, x_1, t = self._latent(x_1=x, x_0=z)

        x_t = (1.0 - t) * x_0 + t * x_1
        v_t = x_1 - x_0

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        return self._reduce(self._loss_fn(self._net(x_t, t), v_t, t))


class GaussianFlowDDPM(GaussianFlowAbstract):
    """Gaussian-source flow on the VP diffusion path (beta schedule)."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.05,
        t_max: float = 0.95,
        cosine_s: float = 0.008,
        base_or_sample: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
    ):
        super().__init__(net=net, dim=dim, n_steps=n_steps, t_min=t_min, t_max=t_max,
                         base_or_sample=base_or_sample, dtype=dtype, device=device)

        self._s0 = float(cosine_s)

    def _vp_coefs(self, t: torch.Tensor):
        s = (1.0 - t).clamp(0.0, 1.0)

        f = ((s + self._s0) / (1.0 + self._s0)) * (math.pi / 2.0)
        f0 = (self._s0 / (1.0 + self._s0)) * (math.pi / 2.0)

        abar = torch.cos(f).pow(2) / (math.cos(f0) ** 2)
        abar = abar.clamp(min=self._eps, max=1.0)

        beta_s = (math.pi / (1.0 + self._s0)) * torch.tan(f)
        beta_s = beta_s.clamp_min(0.0)

        a = torch.sqrt(abar.clamp_min(self._eps))
        sigma = torch.sqrt((1.0 - abar).clamp_min(self._eps))

        return beta_s, abar, a, sigma

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        x_0, x_1, t = self._latent(x_1=x, x_0=z)
        beta_s, abar, a, sigma = self._vp_coefs(t)

        x_t = a * x_1 + sigma * x_0
        v_t = -0.5 * beta_s * (abar * x_t - a * x_1) / (1.0 - abar).clamp_min(self._eps)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        return self._reduce(self._loss_fn(self._net(x_t, t), v_t, t))


class GaussianFlowOT(GaussianFlowAbstract):
    """Gaussian source flow with minibatch Sinkhorn OT coupling + linear path."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.05,
        t_max: float = 0.95,
        sigma_min: float = 1e-5,
        base_or_sample: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
    ):
        super().__init__(net=net, dim=dim, n_steps=n_steps, t_min=t_min, t_max=t_max,
                         base_or_sample=base_or_sample, dtype=dtype, device=device)

        if not (0.0 < sigma_min < 1.0):
            raise ValueError("`sigma_min` must be in (0,1).")

        self._sigma_min = float(sigma_min)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        x_0, x_1, t = self._latent(x_1=x, x_0=z)

        x_t = (1.0 - (1.0 - self._sigma_min) * t) * x_0 + t * x_1
        v_t = x_1 - (1.0 - self._sigma_min) * x_0

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        return self._reduce(self._loss_fn(self._net(x_t, t), v_t, t))
