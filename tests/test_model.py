"""Neural networks module unittests."""

import pytest
import torch
from genkit.nn import MLPModel, timestep_embedding
from genkit.diffusion import DDPMV
from .utils import _devices


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def _ddpm(n_steps=10):
    return DDPMV(net=_ZeroNet(), dim=2, n_steps=n_steps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_timestep_embedding_shape_dtype_device(dtype, device):
    t = torch.rand(17, device=device, dtype=dtype)
    y = timestep_embedding(t, 32).to(dtype=dtype)
    assert y.shape == (17, 32)
    assert y.dtype == dtype
    assert y.device.type == device.type
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(("input_dim", "output_dim", "expected_shape"), [(2, None, (17, 2)), (3, 2, (17, 2))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_mlp_model_shape_dtype_device(input_dim, output_dim, expected_shape, dtype, device):
    model = MLPModel(input_dim=input_dim, output_dim=output_dim, width=16, depth=2, time_dim=8).to(device=device, dtype=dtype)
    x = torch.rand(17, input_dim, device=device, dtype=dtype)
    t = torch.rand(17, 1, device=device, dtype=dtype)
    y = model(x, t)
    assert y.shape == expected_shape
    assert y.dtype == dtype
    assert y.device.type == device.type
    assert torch.isfinite(y).all()


# --- Base._check_t validation ---

def test_check_t_rejects_bool():
    with pytest.raises(TypeError, match="bool"):
        _ddpm()._check_t(True, 4)


def test_check_t_int_below_one_raises():
    with pytest.raises(ValueError, match=r"\[1,"):
        _ddpm(n_steps=10)._check_t(0, 4)


def test_check_t_int_above_n_steps_raises():
    with pytest.raises(ValueError, match=r"\[1,"):
        _ddpm(n_steps=10)._check_t(11, 4)


def test_check_t_float_below_zero_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _ddpm()._check_t(-0.1, 4)


def test_check_t_float_above_one_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _ddpm()._check_t(1.1, 4)


def test_check_t_tensor_wrong_length_raises():
    with pytest.raises(ValueError, match="shape"):
        _ddpm()._check_t(torch.tensor([1, 2, 3]), 4)  # length 3, expected 4


def test_check_t_float_tensor_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _ddpm(n_steps=10)._check_t(torch.tensor([0.5, 0.5, 1.5, 0.5]), 4)


def test_check_t_int_tensor_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[1,"):
        _ddpm(n_steps=10)._check_t(torch.tensor([1, 2, 0, 4], dtype=torch.int32), 4)


def test_check_t_valid_int_returns_broadcast_tensor():
    t = _ddpm(n_steps=10)._check_t(5, 4)
    assert t.shape == (4,)
    assert (t == 5).all()


# --- Base._sample_source with explicit base tensor ---

def test_sample_source_1d_base_expands_to_n_samples():
    base = torch.tensor([1.0, 2.0])
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, base_or_sample=base)
    samples = m._sample_source(5)
    assert samples.shape == (5, 2)
    assert torch.equal(samples, base.unsqueeze(0).expand(5, -1))


def test_sample_source_1d_base_wrong_dim_raises():
    base = torch.tensor([1.0, 2.0, 3.0])  # dim 3, model expects dim 2
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, base_or_sample=base)
    with pytest.raises(ValueError, match="dim"):
        m._sample_source(4)


def test_sample_source_2d_base_samples_from_rows():
    base = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, base_or_sample=base)
    samples = m._sample_source(4)
    assert samples.shape == (4, 2)


def test_sample_source_image_base_expands_to_n_samples():
    base = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    m = DDPMV(net=_ZeroNet(), dim=(1, 4, 4), n_steps=10, base_or_sample=base)
    samples = m._sample_source(3)

    assert samples.shape == (3, 1, 4, 4)
    assert torch.equal(samples, base.unsqueeze(0).expand(3, -1, -1, -1))


def test_ddpm_accepts_image_shaped_batches():
    m = DDPMV(net=_ZeroNet(), dim=(1, 4, 4), n_steps=10)
    x = torch.randn(3, 1, 4, 4)

    loss = m.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert m.sample(2).shape == (2, 1, 4, 4)


def test_ddpm_sigma_max_scales_default_source(monkeypatch):
    def _ones(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _ones)
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=3.0)

    assert torch.equal(m._sample_source_default(4), torch.full((4, 2), 3.0))


def test_ddpm_sigma_max_scales_reverse_posterior_noise(monkeypatch):
    m = DDPMV(net=_ZeroNet(), dim=2, n_steps=2, sigma_max=4.0)
    m._sample_source = lambda n_samples: torch.zeros(n_samples, 2, device=m._device, dtype=m._fdtype)
    m._alpha_bar = torch.full((2,), 0.5, device=m._device, dtype=m._fdtype)
    m._alphas = torch.ones(2, device=m._device, dtype=m._fdtype)
    m._betas = torch.zeros(2, device=m._device, dtype=m._fdtype)
    m._sqrt_post_var = torch.tensor([0.0, 2.0], device=m._device, dtype=m._fdtype)
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)

    samples = m.sample(3)

    assert torch.equal(samples, torch.full((3, 2), 8.0, device=m._device, dtype=m._fdtype))


def test_ddpm_rejects_invalid_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        DDPMV(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=0.0)
