"""Flow module timestep tests."""

import pytest
import torch

from genkit.flow import GaussianFlowLinear


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
