"""Diffusion module indexing tests."""

import pytest
import torch
from genkit.diffusion import DLPMEps


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_dlpmeps_loss_samples_full_time_range(monkeypatch):
    model = DLPMEps(net=_ZeroNet(), dim=2, n_steps=7, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    orig_randint = torch.randint

    def _patched_randint(low, high, size, device=None, **kwargs):
        assert low == 1
        assert high == model._n_steps + 1
        return torch.full(size, high - 1, device=device, dtype=torch.int64)

    monkeypatch.setattr(torch, "randint", _patched_randint)
    loss = model.loss(x)
    monkeypatch.setattr(torch, "randint", orig_randint)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmeps_sample_uses_terminal_sigma_index():
    n_steps = 4
    n_samples = 3
    dim = 2
    model = DLPMEps(net=_ZeroNet(), dim=dim, n_steps=n_steps, device="cpu", fdtype=torch.float32)

    model._sigma_1_t = torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0], dtype=torch.float32)

    model._draw_A = lambda n: torch.ones(n, 1, device=model._device, dtype=model._fdtype)
    model._draw_G = lambda *d: torch.ones(*d, device=model._device, dtype=model._fdtype)

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
    model = DLPMEps(net=_ZeroNet(), dim=2, n_steps=7, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmeps_loss_accepts_normalized_float_t():
    model = DLPMEps(net=_ZeroNet(), dim=2, n_steps=7, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dlpmeps_loss_rejects_invalid_t():
    model = DLPMEps(net=_ZeroNet(), dim=2, n_steps=7, device="cpu", fdtype=torch.float32)
    x = torch.randn(5, 2, dtype=torch.float32)

    with pytest.raises(ValueError):
        model.loss(x, t=0)
