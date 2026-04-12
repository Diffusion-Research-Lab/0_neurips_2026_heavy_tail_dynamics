"""Flow module."""

import numpy as np
import torch
from ._abs import GaussianFlowAbstract


class GaussianFlowLinear(GaussianFlowAbstract):
    """Gaussian source flow with a linear path (Flow Matching)."""
    _family = "flow"
    _loss_tag = "mse"

    def _precompute_loss(self, x, z, t=None):
        """Build linear-path flow targets and corresponding network predictions."""
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)

        x_t = (1.0 - t) * x_0 + t * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        if v_t.shape != v_t_hat.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but v_t_hat has shape"
                             f" {tuple(v_t_hat.shape)}.")

        return v_t_hat, v_t, t

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Return unreduced linear flow-matching losses."""
        v_t_hat, v_t, t = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(v_t_hat, v_t, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced linear flow-matching objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))


class GaussianFlowOT(GaussianFlowAbstract):
    """Gaussian source flow with minibatch Sinkhorn OT coupling + linear path."""
    _family = "flow"
    _loss_tag = "mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.05,
        t_max: float = 0.95,
        sigma_min: float = 1e-5,
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        device: torch.device = "cpu",
    ):
        """Initialize the OT-style Gaussian flow with a minimum variance floor."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, t_min=t_min, t_max=t_max,
                         base_or_sample=base_or_sample, fdtype=fdtype, idtype=idtype,
                         device=device)

        if not (0.0 < sigma_min < 1.0):
            raise ValueError("`sigma_min` must be in (0,1).")

        self._sigma_min = float(sigma_min)

    def _precompute_loss(self, x, z, t=None):
        """Build OT-path flow targets and corresponding network predictions."""
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)

        x_t = (1.0 - (1.0 - self._sigma_min) * t) * x_0 + t * x_1
        v_t = x_1 - (1.0 - self._sigma_min) * x_0
        v_t_hat = self._net(x_t, t)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        if v_t.shape != v_t_hat.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but v_t_hat has shape"
                             f" {tuple(v_t_hat.shape)}.")

        return v_t_hat, v_t, t

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Return unreduced OT flow-matching losses."""
        v_t_hat, v_t, t = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(v_t_hat, v_t, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced OT flow-matching objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))


class GaussianFlowDDPM(GaussianFlowAbstract):
    """Gaussian-source flow on the VP diffusion path (beta schedule)."""
    _family = "flow"
    _loss_tag = "mse"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 1000,
        t_min: float = 0.05,
        t_max: float = 0.95,
        cosine_s: float = 0.008,
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        device: torch.device = "cpu",
    ):
        """Initialize the VP-inspired Gaussian flow and its cosine schedule."""
        super().__init__(net=net, dim=dim, n_steps=n_steps, t_min=t_min, t_max=t_max,
                         base_or_sample=base_or_sample, fdtype=fdtype, idtype=idtype,
                         device=device)

        self._s0 = float(cosine_s)

    def _vp_coefs(self, t: torch.Tensor):
        """Evaluate continuous VP coefficients at the requested flow times."""
        s = (1.0 - t).clamp(0.0, 1.0)

        f = ((s + self._s0) / (1.0 + self._s0)) * (np.pi / 2.0)
        f0 = (self._s0 / (1.0 + self._s0)) * (np.pi / 2.0)

        abar = torch.cos(f).pow(2) / (np.cos(f0) ** 2)
        abar = abar.clamp(min=self._eps, max=1.0)

        beta_s = (np.pi / (1.0 + self._s0)) * torch.tan(f)
        beta_s = beta_s.clamp_min(0.0)

        a = torch.sqrt(abar.clamp_min(self._eps))
        sigma = torch.sqrt((1.0 - abar).clamp_min(self._eps))

        return beta_s, abar, a, sigma

    def _precompute_loss(self, x, z, t=None):
        """Build VP-path velocity targets and network predictions."""
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)

        beta_s, abar, a, sigma = self._vp_coefs(t)
        x_t = a * x_1 + sigma * x_0
        v_t = -0.5 * beta_s * (abar * x_t - a * x_1) / (1.0 - abar).clamp_min(self._eps)
        v_t_hat = self._net(x_t, t)

        if v_t.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        if v_t.shape != v_t_hat.shape:
            raise ValueError(f"Shape mismatch: v_t has shape {tuple(v_t.shape)} but v_t_hat has shape"
                             f" {tuple(v_t_hat.shape)}.")

        return v_t_hat, v_t, t

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Return unreduced VP flow-matching losses."""
        v_t_hat, v_t, t = self._precompute_loss(x=x, z=z, t=t)
        return self._loss_fn(v_t_hat, v_t, t)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: int = None) -> torch.Tensor:
        """Compute the reduced VP flow-matching objective."""
        return self._reduce(self._loss(x=x, z=z, t=t))
