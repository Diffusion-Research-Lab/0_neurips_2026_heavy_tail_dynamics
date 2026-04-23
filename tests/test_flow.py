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
