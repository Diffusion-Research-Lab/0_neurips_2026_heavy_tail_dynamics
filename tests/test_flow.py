"""Flow module timestep tests."""

import pytest
import torch
from genkit.flow import GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT
from .utils import _devices


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def _assert_finite_loss(model_cls, *, t, device, dtype, **kwargs):
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, device=device, fdtype=dtype, **kwargs)
    loss = model.loss(torch.randn(5, 2, dtype=dtype, device=device), t=t)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("t", [3, 0.5])
def test_gaussian_flow_linear_accepts_common_time_formats(t, device, dtype):
    _assert_finite_loss(GaussianFlowLinear, t=t, device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_gaussian_flow_loss_rejects_invalid_t(device, dtype):
    with pytest.raises(ValueError):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, device=device, fdtype=dtype).loss(
            torch.randn(5, 2, dtype=dtype, device=device), t=11
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_other_gaussian_flow_variants_produce_finite_loss(model_cls, kwargs, device, dtype):
    _assert_finite_loss(model_cls, t=0.5, device=device, dtype=dtype, **kwargs)


def test_gaussian_flow_ot_rejects_invalid_sigma_min():
    with pytest.raises(ValueError, match="sigma_min"):
        GaussianFlowOT(net=_ZeroNet(), dim=2, n_steps=10, sigma_min=1.5, device="cpu", fdtype=torch.float32)


@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowLinear, {}), (GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_gaussian_flow_sigma_max_scales_default_source(model_cls, kwargs, monkeypatch):
    def _ones(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _ones)
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=2.5, device="cpu", fdtype=torch.float32, **kwargs)

    assert torch.equal(model._sample_source_default(3), torch.full((3, 2), 2.5))


def test_gaussian_flow_rejects_invalid_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=0.0, device="cpu", fdtype=torch.float32)


@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowLinear, {}), (GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_gaussian_flow_accepts_image_shaped_batches(model_cls, kwargs):
    model = model_cls(net=_ZeroNet(), dim=(1, 4, 4), n_steps=10, device="cpu", fdtype=torch.float32, **kwargs)
    x = torch.randn(3, 1, 4, 4)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.sample(2).shape == (2, 1, 4, 4)


def test_gaussian_flow_check_t_rejects_bool():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(TypeError, match="bool"):
        m._check_t(True, 5)


def test_gaussian_flow_check_t_int_above_n_steps_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError):
        m._check_t(11, 5)


def test_gaussian_flow_check_t_float_tensor_out_of_range_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        m._check_t(torch.tensor([0.5, 1.5, 0.3, 0.4, 0.5]), 5)


def test_gaussian_flow_latent_x1_wrong_dim_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError, match="x1"):
        m._latent(torch.randn(4, 3))  # dim 3, model expects dim 2


def test_gaussian_flow_latent_x0_shape_mismatch_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    x1 = torch.randn(4, 2)
    x0 = torch.randn(4, 3)  # wrong dim
    with pytest.raises(ValueError, match="x0"):
        m._latent(x1, x_0=x0)
