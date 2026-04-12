"""Flow module timestep tests."""

import pytest
import torch

from genkit.flow import GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_gaussian_flow_loss_accepts_integer_t():
    model = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_gaussian_flow_loss_accepts_normalized_float_t():
    model = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_gaussian_flow_loss_rejects_invalid_t():
    model = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    with pytest.raises(ValueError):
        model.loss(x, t=11)


@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_other_gaussian_flow_variants_produce_finite_loss(model_cls, kwargs):
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, device="cpu", fdtype=torch.float32, **kwargs)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_gaussian_flow_ot_rejects_invalid_sigma_min():
    with pytest.raises(ValueError, match="sigma_min"):
        GaussianFlowOT(net=_ZeroNet(), dim=2, n_steps=10, sigma_min=1.5, device="cpu", fdtype=torch.float32)
