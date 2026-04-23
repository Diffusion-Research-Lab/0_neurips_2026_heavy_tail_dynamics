"""Inspect module unittests."""

import importlib.util
import numpy as np
import pytest
import torch
from genkit import DDPMV, DLPMEps, GaussianFlowLinear
from genkit.inspect import (
    estimate_init_error,
    estimate_training_loss_error,
    fit_hmm_on_weight_stats,
    model_est_err_curve,
    model_est_jacobian_spectral_curve,
)
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

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        if z is None:
            z = self._sample_source(len(x))
        if t is None:
            if self._family == "flow":
                t = torch.zeros((len(x), 1), device=self._device, dtype=self._fdtype)
            else:
                t = torch.ones((len(x),), device=self._device, dtype=self._idtype)
        return (x.square().mean() + z.square().mean() + 0.0 * t.to(dtype=x.dtype).mean()).to(dtype=x.dtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        return 3.0 * self._loss(x, z, t).mean()


class _PointwiseLossGenModel(_DummyGenModel):
    """Inspectable generator stub whose native loss is pointwise."""

    def _loss(self, x: torch.Tensor, z: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        return x.square()


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

    curve = model_est_jacobian_spectral_curve(model, x, n_power_iter=4)

    assert curve.shape == (5,)
    assert np.allclose(curve, 2.5)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["flow", "diffusion"])
def test_model_est_jacobian_spectral_curve_caps_time_grid_by_default(device, family):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=32, device=device, dtype=torch.float64, scale=2.5)

    curve = model_est_jacobian_spectral_curve(model, x, n_power_iter=2)

    assert curve.shape == (10,)
    assert np.allclose(curve, 2.5)


@pytest.mark.parametrize("device", _devices())
def test_model_est_jacobian_spectral_curve_validates_arguments(device):
    x = torch.randn(4, 1, device=device, dtype=torch.float32)
    model = _DummyGenModel(family="flow", n_steps=4, device=device, dtype=torch.float32, scale=1.0)

    with pytest.raises(ValueError, match="n_power_iter"):
        model_est_jacobian_spectral_curve(model, x, n_power_iter=0)
    with pytest.raises(ValueError, match="max_n_steps"):
        model_est_jacobian_spectral_curve(model, x, max_n_steps=0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(("family",), [("flow",), ("diffusion",)])
def test_model_est_err_curve_returns_native_grid(device, family):
    x = torch.tensor([[1.0], [-2.0], [0.5]], device=device, dtype=torch.float64)
    model = _DummyGenModel(family=family, n_steps=4, device=device, dtype=torch.float64, scale=1.0)

    curve = model_est_err_curve(model, x)

    assert curve.shape == (4,)
    assert np.all(curve >= 0.0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["flow", "diffusion"])
def test_model_est_err_curve_reduces_pointwise_loss(device, family):
    x = torch.tensor([[1.0, -2.0], [0.5, 1.5]], device=device, dtype=torch.float64)
    model = _PointwiseLossGenModel(family=family, n_steps=12, device=device, dtype=torch.float64, scale=1.0)

    curve = model_est_err_curve(model, x)

    assert curve.shape == (10,)
    assert np.allclose(curve, float(model.loss(x).item()))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (GaussianFlowLinear, {}),
        (DDPMV, {}),
        (DLPMEps, {"alpha": 1.6}),
    ],
)
def test_model_est_err_curve_supports_native_and_mse_loss(device, model_cls, kwargs):
    x = torch.randn(8, 1, device=device, dtype=torch.float64)
    net = _LinearTimeNet(scale=1.0).to(device=device, dtype=torch.float64)
    model = model_cls(net=net, dim=1, n_steps=8, fdtype=torch.float64, device=device, **kwargs)

    native_curve = model_est_err_curve(model, x, max_n_steps=5, loss_type="native")
    mse_curve = model_est_err_curve(model, x, max_n_steps=5, loss_type="mse")

    assert native_curve.shape == (5,)
    assert mse_curve.shape == (5,)
    assert np.all(native_curve >= 0.0)
    assert np.all(mse_curve >= 0.0)


def test_model_est_err_curve_rejects_unknown_loss_type():
    x = torch.randn(8, 1, dtype=torch.float64)
    net = _LinearTimeNet(scale=1.0).to(dtype=torch.float64)
    model = GaussianFlowLinear(net=net, dim=1, n_steps=8, fdtype=torch.float64, device="cpu")

    with pytest.raises(ValueError, match="loss_type"):
        model_est_err_curve(model, x, loss_type="other")


def test_fit_hmm_on_weight_stats_requires_hmmlearn():
    if importlib.util.find_spec("hmmlearn") is not None:
        pytest.skip("hmmlearn is installed in this environment")

    with pytest.raises(ModuleNotFoundError, match="hmmlearn"):
        fit_hmm_on_weight_stats(np.ones((4, 2)))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (GaussianFlowLinear, {}),
        (DDPMV, {}),
        (DLPMEps, {"alpha": 1.6}),
    ],
)
def test_estimate_training_loss_error_supports_native_and_mse_loss(device, model_cls, kwargs):
    x = torch.randn(8, 1, device=device, dtype=torch.float64)
    net = _LinearTimeNet(scale=1.0).to(device=device, dtype=torch.float64)
    model = model_cls(net=net, dim=1, n_steps=8, fdtype=torch.float64, device=device, **kwargs)

    mse = estimate_training_loss_error(model, x, n_batches=2, batch_size=4, loss_type="mse")
    native = estimate_training_loss_error(model, x, n_batches=2, batch_size=4, loss_type="native")

    assert native >= 0.0
    assert mse >= 0.0


def test_estimate_training_loss_error_rejects_unknown_loss_type():
    x = torch.randn(8, 1, dtype=torch.float64)
    net = _LinearTimeNet(scale=1.0).to(dtype=torch.float64)
    model = GaussianFlowLinear(net=net, dim=1, n_steps=8, fdtype=torch.float64, device="cpu")

    with pytest.raises(ValueError, match="loss_type"):
        estimate_training_loss_error(model, x, loss_type="other")


def test_estimate_init_error_requires_source_sampler_for_models_without_source_method():
    class _NoSourceModel:
        _family = "flow"
        _device = "cpu"
        _fdtype = torch.float32
        _idtype = torch.int64
        _n_steps = 4

        def __init__(self):
            self._net = _LinearTimeNet(scale=1.0).to(dtype=torch.float32)

    model = _NoSourceModel()
    x = torch.randn(16, 1)

    with pytest.raises(ValueError, match="source_sampler"):
        estimate_init_error(model, x)
