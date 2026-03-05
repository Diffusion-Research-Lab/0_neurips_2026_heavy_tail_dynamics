"""Utils module unittests."""

import pytest
import math
import torch
from cauda.utils import cosine_betas, make_schedule
from .utils import _devices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_betas_shape_dtype_device(dtype, device):
    betas = cosine_betas(n_steps=17, device=device, dtype=dtype, s=0.008)
    assert betas.shape == (17,)
    assert betas.dtype == dtype
    assert betas.device.type == device.type


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_betas_first_is_zero(dtype, device):
    betas = cosine_betas(n_steps=50, device=device, dtype=dtype, s=0.008)
    assert torch.allclose(betas[0], torch.zeros((), device=device, dtype=dtype), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_betas_finite_and_range(dtype, device):
    betas = cosine_betas(n_steps=200, device=device, dtype=dtype, s=0.008)
    assert torch.isfinite(betas).all()
    # Expected in [0, 1); allow tiny numerical slack.
    assert (betas >= -1e-12).all()
    assert (betas < 1.0 + 1e-12).all()
    assert betas.std() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_cosine_betas_matches_definition(dtype, device):
    n_steps = 123
    s = 0.01
    betas = cosine_betas(n_steps=n_steps, device=device, dtype=dtype, s=s)

    t = torch.arange(0, n_steps, device=device, dtype=dtype)
    alpha_bar = torch.cos(((t / n_steps) + s) / (1.0 + s) * (math.pi / 2.0)).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha_bar_prev = torch.cat([alpha_bar[:1], alpha_bar[:-1]], dim=0)
    ref = 1.0 - alpha_bar / alpha_bar_prev

    assert torch.allclose(betas, ref, atol=0.0, rtol=0.0)


def test_cosine_betas_n_steps_zero_raises():
    with pytest.raises((RuntimeError, IndexError)):
        _ = cosine_betas(n_steps=0, device="cpu", dtype=torch.float32, s=0.008)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_make_schedule_shapes_and_identities(dtype, device):
    n = 37
    betas, alphas, alpha_bar = make_schedule(n_steps=n, device=device, dtype=dtype, s=0.008)

    assert betas.shape == (n,)
    assert alphas.shape == (n,)
    assert alpha_bar.shape == (n + 1,)

    assert betas.dtype == dtype and alphas.dtype == dtype and alpha_bar.dtype == dtype
    assert betas.device.type == device.type
    assert alphas.device.type == device.type
    assert alpha_bar.device.type == device.type

    assert torch.allclose(alphas, 1.0 - betas, atol=0.0, rtol=0.0)
    assert torch.allclose(alpha_bar[0], torch.tensor(1.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
    assert torch.allclose(alpha_bar[1:], torch.cumprod(alphas, dim=0), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_make_schedule_alpha_bar_monotone(dtype, device):
    n = 200
    _, alphas, alpha_bar = make_schedule(n_steps=n, device=device, dtype=dtype, s=0.008)

    assert torch.isfinite(alpha_bar).all()
    assert (alphas > 0).all()  # since betas in [0,1)
    assert (alphas <= 1).all()

    # alpha_bar in (0,1] and non-increasing
    assert (alpha_bar <= 1.0).all()
    assert (alpha_bar > 0.0).all()
    diffs = alpha_bar[1:] - alpha_bar[:-1]
    assert (diffs <= 1e-12).all()
