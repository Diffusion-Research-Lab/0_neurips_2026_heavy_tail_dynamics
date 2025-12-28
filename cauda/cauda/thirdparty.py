"""Wrapper module for https://github.com/darioShar/DLPM/."""

# Authors: Hamza Cherkaoui

import os
import sys
import warnings
from pathlib import Path
from typing import Optional
import torch
from ._abs import Base


def _resolve_dlpm_authors_root(authors_root: Optional[str] = None) -> str:
    if authors_root is None:
        authors_root = os.environ.get("DLPM_AUTHORS_ROOT", None)
    if authors_root is None:
        raise RuntimeError(
            "Set DLPM_AUTHORS_ROOT to the authors repo root that contains the `dlpm/` package "
            "(e.g. .../DLPM/DLPM)."
        )
    root = Path(authors_root).expanduser().resolve()
    if not (root / "dlpm").is_dir():
        raise RuntimeError(f"Invalid DLPM_AUTHORS_ROOT={root} (missing `dlpm/` directory),"
                           f" download it from 'https://github.com/darioShar/DLPM'.")
    return str(root)


def _ensure_on_syspath(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


class _NetAdapter(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        p = next(self.net.parameters(), None)
        if p is not None:
            if x.device != p.device:
                x = x.to(p.device)
            if t.device != p.device:
                t = t.to(p.device)
            if x.dtype != p.dtype:
                x = x.to(p.dtype)
            if t.dtype != p.dtype:
                t = t.to(p.dtype)

        return self.net(x, t)


class DLPMEpsOrigin(Base):
    """
    Adapter around the authors' GenerativeLevyProcess(DLPM).
    """

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        alpha: float = 1.8,
        authors_root: Optional[str] = None,
        time_spacing: str = "linear",
        rescale_timesteps: bool = True,
        isotropic: bool = True,
        loss_monte_carlo: bool = 'mean',
        monte_carlo_outer: int = 5,
        monte_carlo_inner: int = 1,
        lploss: float = 2.0,
        clamp_a: Optional[float] = None,
        clamp_eps: Optional[float] = None,
        scale: str = "scale_preserving",
        base_or_sample: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         dtype=dtype, device=device)

        root = _resolve_dlpm_authors_root(authors_root)
        _ensure_on_syspath(root)

        from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess  # noqa: E402

        self._net_auth = _NetAdapter(self._net)
        self._loss_monte_carlo = loss_monte_carlo
        self._monte_carlo_outer = monte_carlo_outer
        self._monte_carlo_inner = monte_carlo_inner
        self._lploss = lploss
        self._clamp_a = clamp_a
        self._clamp_eps = clamp_eps

        self._glp = GenerativeLevyProcess(alpha=float(alpha),
                                          device=self._device,
                                          reverse_steps=int(n_steps),
                                          time_spacing=str(time_spacing),
                                          rescale_timesteps=bool(rescale_timesteps),
                                          isotropic=bool(isotropic),
                                          scale=str(scale))

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:

        if z is not None:
            warnings.warn("In 'DLPMEpsOrigin.loss', input 'z' is ignored (z = A G are "
                          "sampled internally).")

        x = x.to(device=self._device, dtype=self._dtype)
        out = self._glp.training_losses(models={"default": self._net_auth},
                                        x_start=x,
                                        loss_type="EPS_LOSS",
                                        lploss=self._lploss,
                                        loss_monte_carlo=self._loss_monte_carlo,
                                        monte_carlo_outer=self._monte_carlo_outer,
                                        monte_carlo_inner=self._monte_carlo_inner,
                                        clamp_a=self._clamp_a,
                                        clamp_eps=self._clamp_eps)

        return out["loss"].to(dtype=self._dtype)

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        self._net.eval()

        if self._clamp_a is not None:
            self._glp.dlpm.gen_a.setParams(clamp_a=self._clamp_a)

        if self._clamp_eps is not None:
            self._glp.dlpm.gen_eps.setParams(clamp_eps=self._clamp_eps)

        x = self._glp.p_sample_loop(model=self._net_auth,
                                    shape=(int(n_samples), int(self._dim)),
                                    progress=bool(kwargs.get("progress", False)))

        return x.to(device=self._device, dtype=self._dtype)
