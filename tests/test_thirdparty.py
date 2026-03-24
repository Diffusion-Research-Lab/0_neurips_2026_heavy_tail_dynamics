"""Tests for optional third-party adapters."""

from pathlib import Path
import pytest
import torch
from genkit.nn import MLPModel
from genkit.thirdparty import DLPMEpsOrigin, FlowMatchingOrigin, ScoreSDEOrigin


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_flow_matching_vendor(tmp_path: Path) -> Path:
    root = tmp_path / "flow_matching_vendor"
    _write(root / "flow_matching" / "__init__.py", "")
    _write(
        root / "flow_matching" / "path" / "__init__.py",
        """
import torch
from types import SimpleNamespace


class AffineProbPath:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, x_0, x_1, t):
        x_t = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        dx_t = x_1 - x_0
        return SimpleNamespace(x_t=x_t, dx_t=dx_t, t=t, x_0=x_0, x_1=x_1)
""",
    )
    _write(
        root / "flow_matching" / "path" / "scheduler.py",
        """
class CondOTScheduler:
    pass
""",
    )
    _write(
        root / "flow_matching" / "solver" / "__init__.py",
        """
class ODESolver:
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model

    def sample(self, x_init, step_size, method, time_grid, return_intermediates=False, **kwargs):
        x = x_init
        t = time_grid[-1].expand(x.size(0))
        return x + (time_grid[-1] - time_grid[0]) * self.velocity_model(x=x, t=t)
""",
    )
    _write(
        root / "flow_matching" / "utils" / "__init__.py",
        """
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
""",
    )
    return root


def _make_score_sde_vendor(tmp_path: Path) -> Path:
    root = tmp_path / "score_sde_vendor"
    _write(
        root / "sde_lib.py",
        """
import torch


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=100):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.N = int(N)

    def marginal_prob(self, x, t):
        std = torch.full((x.size(0),), self.sigma_min, device=x.device, dtype=x.dtype)
        return x, std

    def prior_sampling(self, shape):
        return self.sigma_max * torch.randn(*shape)

    def sde(self, x, t):
        drift = torch.zeros_like(x)
        diffusion = torch.full((x.size(0),), self.sigma_max, device=x.device, dtype=x.dtype)
        return drift, diffusion
""",
    )
    _write(
        root / "losses.py",
        """
def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=False, eps=1e-3):
    def loss_fn(model, batch):
        out = model(batch, batch.new_zeros(batch.size(0)))
        return out.square().mean()
    return loss_fn
""",
    )
    _write(
        root / "sampling.py",
        """
import torch


class ReverseDiffusionPredictor:
    pass


class NoneCorrector:
    pass


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cpu'):
    def sampler(model):
        x = torch.zeros(shape, device=device)
        return inverse_scaler(x), sde.N
    return sampler
""",
    )
    return root


def _make_dlpm_vendor(tmp_path: Path) -> Path:
    root = tmp_path / "DLPM"
    _write(root / "dlpm" / "__init__.py", "")
    _write(root / "dlpm" / "methods" / "__init__.py", "")
    _write(
        root / "dlpm" / "methods" / "GenerativeLevyProcess.py",
        """
from types import SimpleNamespace
import torch


class _Params:
    def setParams(self, **kwargs):
        self.kwargs = dict(kwargs)


class GenerativeLevyProcess:
    def __init__(self, alpha, device, reverse_steps, time_spacing, rescale_timesteps, isotropic, scale):
        self.dlpm = SimpleNamespace(gen_a=_Params(), gen_eps=_Params())

    def training_losses(self, models, x_start, **kwargs):
        return {"loss": x_start.square().mean()}

    def p_sample_loop(self, model, shape, progress=False):
        return torch.zeros(shape)
""",
    )
    return root


@pytest.mark.parametrize("fdtype", [torch.float32])
def test_flow_matching_origin_loss_and_sample(tmp_path, fdtype):
    root = _make_flow_matching_vendor(tmp_path)
    net = MLPModel(dim=2, width=8, depth=1)
    model = FlowMatchingOrigin(net=net, dim=2, n_steps=4, package_root=str(root), fdtype=fdtype)
    x = torch.randn(6, 2, dtype=fdtype)

    loss = model.loss(x)
    sample = model.sample(5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert sample.shape == (5, 2)
    assert sample.dtype == fdtype


def test_score_sde_origin_loss_and_sample(tmp_path):
    root = _make_score_sde_vendor(tmp_path)
    net = MLPModel(dim=2, width=8, depth=1)
    model = ScoreSDEOrigin(net=net, dim=2, n_steps=4, package_root=str(root), fdtype=torch.float32)
    x = torch.randn(6, 2, dtype=torch.float32)

    loss = model.loss(x)
    sample = model.sample(5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert sample.shape == (5, 2)


def test_dlpmeps_origin_still_accepts_legacy_dtype_alias(tmp_path):
    root = _make_dlpm_vendor(tmp_path)
    net = MLPModel(dim=2, width=8, depth=1)
    model = DLPMEpsOrigin(net=net, dim=2, n_steps=4, authors_root=str(root), dtype=torch.float32)
    x = torch.randn(6, 2, dtype=torch.float32)

    loss = model.loss(x)
    sample = model.sample(5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert sample.shape == (5, 2)
