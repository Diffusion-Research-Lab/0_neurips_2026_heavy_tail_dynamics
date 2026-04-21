"""Diffusion module indexing tests."""

import pytest
import torch
from genkit.diffusion import DLPMEps, DLPMEpsC
from genkit.nn import MLPModel


class _ZeroNet(torch.nn.Module):
    """Minimal network that always predicts zero noise."""

    def forward(self, x, t):
        """Return a zero tensor with the same shape as the input batch."""
        return torch.zeros_like(x)


def _make_dlpmeps(n_steps: int = 7) -> DLPMEps:
    """Build one small CPU DLPM model for indexing-focused tests."""
    return DLPMEps(net=_ZeroNet(), dim=2, n_steps=n_steps, device="cpu", fdtype=torch.float32)


def test_dlpmeps_loss_samples_vendor_time_range(monkeypatch):
    model = _make_dlpmeps(n_steps=7)
    x = torch.randn(5, 2, dtype=torch.float32)

    def _patched_randint(low, high, size, device=None, **kwargs):
        assert low == 1
        assert high == model._n_steps
        return torch.full(size, high - 1, device=device, dtype=torch.int64)

    monkeypatch.setattr(torch, "randint", _patched_randint)
    loss = model.loss(x)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmeps_sample_uses_vendor_terminal_sigma_index(monkeypatch):
    n_steps = 4
    n_samples = 3
    dim = 2
    model = _make_dlpmeps(n_steps=n_steps)

    model._sigma_1_t = torch.tensor([0.0, 0.0, 0.0, 7.0, 99.0], dtype=torch.float32)
    model._draw_A = lambda n: torch.ones(n, 1, device=model._device, dtype=model._fdtype)

    def _patched_randn(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _patched_randn)

    def _neutral_reverse(Sigma_1_t, t):
        sigma_hat = torch.zeros(n_samples, device=model._device, dtype=model._fdtype)
        gamma_t = torch.tensor(1.0, device=model._device, dtype=model._fdtype)
        Gamma_t = torch.zeros(n_samples, device=model._device, dtype=model._fdtype)
        return sigma_hat, gamma_t, Gamma_t

    model._g_Sigma_hat_Gamma = _neutral_reverse

    out = model.sample(n_samples=n_samples)
    expected = torch.full((n_samples, dim), 7.0, dtype=model._fdtype, device=model._device)
    assert torch.allclose(out, expected, atol=2e-3, rtol=0.0)


def test_dlpmeps_loss_accepts_integer_t():
    model = _make_dlpmeps(n_steps=7)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmeps_loss_accepts_normalized_float_t():
    model = _make_dlpmeps(n_steps=7)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmeps_loss_rejects_invalid_t():
    model = _make_dlpmeps(n_steps=7)
    x = torch.randn(5, 2, dtype=torch.float32)

    with pytest.raises(ValueError):
        model.loss(x, t=0)


def test_dlpmepsc_loss_accepts_conditioned_mlp_model():
    net = MLPModel(input_dim=3, output_dim=2, width=8, depth=1, time_dim=8)
    model = DLPMEpsC(net=net, dim=2, n_steps=7, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmepsc_rejects_misconfigured_mlp_model():
    net = MLPModel(input_dim=2, width=8, depth=1, time_dim=8)
    with pytest.raises(ValueError, match="net.input_dim=3"):
        DLPMEpsC(net=net, dim=2, n_steps=7, device="cpu", fdtype=torch.float32)
