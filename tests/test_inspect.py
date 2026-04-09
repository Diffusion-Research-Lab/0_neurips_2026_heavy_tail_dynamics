"""Inspect module unittests."""

import importlib.util
import numpy as np
import pytest
import torch

from genkit.inspect import fit_hmm_on_weight_stats, model_est_err_curve, model_est_jacobian_spectral_curve
from .utils import _devices


class _LinearTimeNet(torch.nn.Module):
    """Simple linear net with a known Lipschitz constant."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * x + 0.0 * t


class _DummyGenModel:
    """Minimal inspectable generator stub."""

    def __init__(self, family: str, n_steps: int, device: str, dtype: torch.dtype, scale: float) -> None:
        self._family = family
        self._n_steps = int(n_steps)
        self._fdtype = dtype
        self._idtype = torch.int64
        self._device = device
        self._net = _LinearTimeNet(scale=scale).to(device=device, dtype=dtype)

    def _sample_source(self, n_samples: int) -> torch.Tensor:
        return torch.zeros((n_samples, 1), device=self._device, dtype=self._fdtype)

    def _loss(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (x.square().mean() + z.square().mean() + 0.0 * t.to(dtype=x.dtype)).to(dtype=x.dtype)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("family", "expected_t"),
    [
        ("flow", np.arange(5)),
        ("diffusion", np.arange(1, 6)),
    ],
)
def test_model_est_jacobian_spectral_curve_matches_linear_constant(device, family, expected_t):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=5, device=device, dtype=torch.float64, scale=2.5)

    curve, t_grid = model_est_jacobian_spectral_curve(model, x, n_power_iter=4)

    assert curve.shape == (5,)
    assert t_grid.shape == (5,)
    assert np.allclose(curve, 2.5)
    assert np.allclose(t_grid, expected_t)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["flow", "diffusion"])
def test_model_est_jacobian_spectral_curve_caps_time_grid_by_default(device, family):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=32, device=device, dtype=torch.float64, scale=2.5)

    curve, t_grid = model_est_jacobian_spectral_curve(model, x, n_power_iter=2)

    assert curve.shape == (10,)
    assert t_grid.shape == (10,)
    assert np.allclose(curve, 2.5)
    assert t_grid[0] == pytest.approx(0.0 if family == "flow" else 1.0)


@pytest.mark.parametrize("device", _devices())
def test_model_est_jacobian_spectral_curve_validates_arguments(device):
    x = torch.randn(4, 1, device=device, dtype=torch.float32)
    model = _DummyGenModel(family="flow", n_steps=4, device=device, dtype=torch.float32, scale=1.0)

    with pytest.raises(ValueError, match="n_power_iter"):
        model_est_jacobian_spectral_curve(model, x, n_power_iter=0)
    with pytest.raises(ValueError, match="max_n_steps"):
        model_est_jacobian_spectral_curve(model, x, max_n_steps=0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("family", "expected_t"),
    [
        ("flow", np.linspace(0.0, 1.0, 4)),
        ("diffusion", np.arange(1, 5)),
    ],
)
def test_model_est_err_curve_returns_native_grid(device, family, expected_t):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=4, device=device, dtype=torch.float64, scale=1.0)

    curve, t_grid = model_est_err_curve(model, x)

    assert curve.shape == (4,)
    assert t_grid.shape == (4,)
    assert np.all(curve >= 0.0)
    assert np.allclose(t_grid, expected_t)


def test_fit_hmm_on_weight_stats_requires_hmmlearn():
    if importlib.util.find_spec("hmmlearn") is not None:
        pytest.skip("hmmlearn is installed in this environment")

    with pytest.raises(ModuleNotFoundError, match="hmmlearn"):
        fit_hmm_on_weight_stats(np.ones((4, 2)))
