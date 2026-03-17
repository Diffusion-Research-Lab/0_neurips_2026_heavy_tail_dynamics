"""Diffusion module."""

import warnings
from typing import Tuple
import torch
from ._abs import Base, DDPMAbstarct
from ._sampling import sample_scaled_scalar_alpha_stable
from .utils import cosine_schedule


class DDPMEps(DDPMAbstarct):
    """DDPM (epsilon/noise prediction) with a fixed beta schedule (variance-preserving)."""

    def _loss_fn(self, eps_hat: torch.Tensor, eps: torch.Tensor, t: int):
        return torch.nn.functional.mse_loss(eps_hat, eps, reduction='none')

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:

        _, x_t, eps, t_norm, t, _ = self._latent(x_1=x, eps=z)

        eps_hat = self._net(x_t, t_norm)
        if eps.shape != x_t.shape:
            raise ValueError(f"Shape mismatch: x0 has shape {tuple(eps.shape)} but x_t has shape"
                             f" {tuple(x_t.shape)}.")

        return self._reduce(self._loss_fn(eps_hat, eps, t))

    def _get_eps_hat(self, x, t_norm, t_idx):
        return self._net(x, t_norm)


class DDPMX0(DDPMAbstarct):
    """DDPM with x0-prediction parameterization."""

    def _loss_fn(self, x0_hat: torch.Tensor, x0: torch.Tensor, t: int) -> torch.Tensor:
        alpha_bar = self._alpha_bar.index_select(0, t)
        w = (alpha_bar / (1.0 - alpha_bar)).view(-1, 1).clamp(min=1e-3, max=1e3)
        return w * (x0_hat - x0).square()

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        x_1, x_t, _, t_norm, t, _ = self._latent(x_1=x, eps=z)

        x_1_hat = self._net(x_t, t_norm)
        if x_1_hat.shape != x_1.shape:
            raise ValueError(f"Shape mismatch: x_1_hat={tuple(x_1_hat.shape)} vs"
                             f" x_1={tuple(x_1.shape)}")

        return self._reduce(self._loss_fn(x_1_hat, x_1, t))

    def _get_eps_hat(self, x, t_norm, t_idx):
        return (x - torch.sqrt(self._alpha_bar[t_idx]) * self._net(x, t_norm)) / torch.sqrt(1.0 - self._alpha_bar[t_idx])


class DLPMEps(Base):
    """DLPM (epsilon/noise prediction) with a fixed beta schedule (variance-preserving)."""

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        alpha: float = 1.8,
        n_trial_A: int = 5,
        n_trial_G: int = 1,
        reduce_type: str = "mean",
        clamp_A: Tuple[float, float] = (0.0, 1e3),
        clamp_eps: Tuple[float, float] = None,
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        device: torch.device = 'cpu',
    ):
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)

        self._a = float(alpha)
        if not (0.0 < self._a < 2.0):
            raise ValueError(f"'alpha' must be in (0,2), got {self._a}.")

        _, _, self._betas, _ = cosine_schedule(n_steps, device=device, fdtype=fdtype, idtype=idtype)
        self._gamma_t = (1.0 - self._betas).clamp_min(self._eps).pow(1.0 / self._a)
        self._sigma_t = (1.0 - self._gamma_t.pow(self._a)).clamp_min(0.0).pow(1.0 / self._a)
        self._gamma_1_t = torch.ones(self._n_steps + 1, device=self._device, dtype=self._fdtype)
        self._gamma_1_t[1:] = torch.cumprod(self._gamma_t, dim=0)
        self._sigma_1_t = (1.0 - self._gamma_1_t.pow(self._a)).clamp_min(0.0).pow(1.0 / self._a)

        self._n_trial_A = int(n_trial_A)
        self._n_trial_G = int(n_trial_G)

        if reduce_type not in ("mean", "median"):
            raise ValueError(f"'reduce_type' must be in ('mean','median'), got {reduce_type}.")
        self._reduce_type = reduce_type

        self._clamp_A = clamp_A
        if (self._clamp_A is not None):
            if not (isinstance(self._clamp_A, (tuple, list)) and len(self._clamp_A) == 2):
                raise ValueError("clamp_A must be a (min,max) tuple or None.")
            self._A_min, self._A_max = float(self._clamp_A[0]), float(self._clamp_A[1])

        self._clamp_eps = clamp_eps
        if (self._clamp_eps is not None):
            if not (isinstance(self._clamp_eps, (tuple, list)) and len(self._clamp_eps) == 2):
                raise ValueError("clamp_eps must be a (min,max) tuple/list or None.")
            self._eps_min, self._eps_max = float(self._clamp_eps[0]), float(self._clamp_eps[1])

    def _safe_A(self, A):
        A = A.clamp_min(self._eps)  # by default
        if self._clamp_A is not None:
            return A.clamp(min=self._A_min, max=self._A_max)
        else:
            return A

    def _safe_eps(self, eps):
        if self._clamp_eps is not None:
            return eps.clamp(min=self._eps_min, max=self._eps_max)
        else:
            return eps

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError("In 'DLPMEps', '_sample_source_default' should not be used.")

    def _sample_source(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError("In 'DLPMEps', '_sample_source' should not be used.")

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        loss_values = loss_values.reshape(self._n_trial_A, self._n_trial_G, self._n)
        inner_rms = loss_values.mean(dim=1).clamp_min(self._eps).sqrt()
        if self._reduce_type == "mean":
            return inner_rms.mean()
        return inner_rms.median(dim=0).values.mean()

    def _loss_fn(self, eps_hat: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        return (eps_hat - eps).square().sum(dim=-1)

    def _Sigma_1_t(self, A: torch.Tensor) -> torch.Tensor:
        S = torch.zeros((self._n_steps + 1, A.shape[1]), device=A.device, dtype=A.dtype)
        for t in range(1, self._n_steps + 1):
            S[t] = self._sigma_t[t - 1].square() * A[t - 1]
            S[t] += self._gamma_t[t - 1].square() * S[t - 1]
        return S

    def _g_Sigma_hat_Gamma(self, Sigma_1_t: torch.Tensor, t: int):
        Sigma_ratio = Sigma_1_t[t - 1] / Sigma_1_t[t].clamp_min(self._eps)
        Gamma_t = 1.0 - Sigma_ratio * self._gamma_t[t - 1].square()
        Gamma_t = Gamma_t.clamp(0.0, 1.0)
        Sigma_hat = (Gamma_t * Sigma_1_t[t - 1]).clamp_min(0.0)
        return Sigma_hat, self._gamma_t[t - 1], Gamma_t

    def _expand(self, x):  # since we stack the Monte Carlo drawing in the different dimension
        return x.expand(self._n_trial_A, self._n_trial_G, self._n)

    def _normalize_t(self, t):
        return (t - 1).to(self._fdtype) / float(max(self._n_steps - 1, 1))

    def _draw_G(self, *d):
        return torch.randn(*d, device=self._device, dtype=self._fdtype)

    def _draw_A(self, n):
        A = sample_scaled_scalar_alpha_stable(n_samples=n, alpha=self._a, device=self._device,
                                              dtype=self._fdtype)
        return self._safe_A(A)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        if x_1.ndim != 2 or x_1.size(1) != self._dim:
            raise ValueError(f"Expected x shape (N,{self._dim}), got {tuple(x.shape)}")
        self._n = x_1.size(0)

        if z is not None:
            warnings.warn("In 'DLPMEps.loss', input 'z' is ignored "
                          "(z = A G are sampled internally).")

        # draw time (shared across Monte Carlo replicates for each data point)
        t = torch.randint(1, self._n_steps + 1, (self._n,), device=self._device)  # (n,)
        t_e = self._expand(t.view(1, 1, self._n)).reshape(-1)  # (_n_trial_A * _n_trial_G * n,)
        t_norm = self._expand(self._normalize_t(t).view(1, 1, self._n)).reshape(-1, 1)  # (_n_trial_A * _n_trial_G * n, 1)

        # compute eps the isotropic alpha stable noise
        A = self._draw_A(self._n_trial_A * self._n)
        A = self._expand(A.view(self._n_trial_A, 1, self._n)).reshape(-1)  # (_n_trial_A * _n_trial_G * n,)
        G = self._draw_G(self._n_trial_A, self._n_trial_G, self._n, self._dim)
        G = G.reshape(-1, self._dim)  # (_n_trial_A * _n_trial_G * n, dim)
        eps = A.sqrt().unsqueeze(-1) * G  # (_n_trial_A * _n_trial_G * n, dim)

        gamma_1_t = self._gamma_1_t.index_select(0, t_e).unsqueeze(-1)  # (_n_trial_A * _n_trial_G * n, 1)
        sigma_1_t = self._sigma_1_t.index_select(0, t_e).unsqueeze(-1)  # (_n_trial_A * _n_trial_G * n, 1)
        x_1_e = x_1.view(1, 1, self._n, self._dim).expand(self._n_trial_A, self._n_trial_G, self._n, self._dim)
        x_1_e = x_1_e.reshape(-1, self._dim)

        x_t = gamma_1_t * x_1_e + sigma_1_t * eps

        eps_hat = self._net(x_t, t_norm)
        return self._reduce(self._loss_fn(eps_hat, eps, t))

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        self._net.eval()

        # Sample latent stable path A_{1:T} used to build Sigma_{1->t}(A_{1:t})
        A_path = self._draw_A(self._n_steps * n_samples)
        A_path = self._safe_A(A_path).reshape(self._n_steps, n_samples)
        Sigma_1_t = self._Sigma_1_t(A_path)

        # x_T = \bar \sigma_{T} sqrt(A0) G0
        A0 = self._draw_A(n_samples).squeeze(-1)
        G0 = self._draw_G(n_samples, self._dim)
        eps0 = A0.sqrt().unsqueeze(-1) * G0
        x_t = self._sigma_1_t[self._n_steps] * eps0

        # Reverse recursion (Table 4 DLPM): mean update divided by gamma_t, then add Gaussian innovation
        for t in range(self._n_steps, 0, -1):
            t_norm = torch.full((n_samples, 1), (t - 1) / float(max(self._n_steps - 1, 1)), device=self._device, dtype=self._fdtype)

            Sigma_hat, gamma_t, Gamma_t = self._g_Sigma_hat_Gamma(Sigma_1_t, t)

            x_t = x_t / gamma_t.clamp_min(self._eps) - Gamma_t.unsqueeze(-1) * self._sigma_1_t[t] * self._net(x_t, t_norm)

            if t > 1:
                G = self._draw_G(n_samples, self._dim)
                x_t = x_t + Sigma_hat.clamp_min(self._eps).sqrt().unsqueeze(-1) * G

        return x_t
