"""Metrics module unittests."""

# Authors: Hamza Cherkaoui

import math
import pytest
import torch
from cauda.metrics import (msle_at_quantile, fid, wasserstein2_1d, sliced_wasserstein2, mmd_rbf,
                           mmd_imq, f1)
from .utils import _devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_msle_at_quantile_scaling_closed_form(device, dtype):
    torch.manual_seed(0)
    n = 4096
    xi = 0.9
    c = 3.5

    x = torch.exp(torch.randn(n, device=device, dtype=dtype))  # strictly positive
    y = c * x

    v = msle_at_quantile(x, y, xi=xi, n_grid=256, eps=1e-12)
    expected = (1.0 - xi) * (math.log(c) ** 2)
    expected_t = torch.tensor(expected, device=device, dtype=dtype)

    assert v.shape == ()
    assert torch.isfinite(v)
    assert torch.allclose(v, expected_t, rtol=3e-2, atol=2e-3)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fid_gaussian_mean_shift_matches_theory(device, dtype):
    torch.manual_seed(0)
    n, d = 4096, 6

    mu = torch.tensor([1.0, -0.5, 0.25, 0.0, 0.75, -1.25], device=device, dtype=dtype)
    x = torch.randn(n, d, device=device, dtype=dtype)
    y = torch.randn(n, d, device=device, dtype=dtype) + mu

    v = fid(x, y)
    expected = (mu * mu).sum()
    rel_err = (v - expected).abs() / expected.clamp_min(torch.finfo(dtype).eps)

    assert v.shape == ()
    assert torch.isfinite(v)
    assert v >= 0
    assert rel_err.item() < 0.15  # empirical covariances => tolerate finite-sample error


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein2_translation_formula(device, dtype):
    torch.manual_seed(0)
    n, d = 512, 4
    n_proj = 256

    x = torch.randn(n, d, device=device, dtype=dtype)
    c = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    y = x + c  # exact translation, same sample pairing

    sw2_sq = sliced_wasserstein2(x, y, n_projections=n_proj, sqrt=False, seed=123)
    expected = (c @ c) / float(d)  # E_theta[<theta,c>^2] = ||c||^2 / d

    assert sw2_sq.shape == ()
    assert torch.isfinite(sw2_sq)
    assert sw2_sq >= 0
    assert torch.allclose(sw2_sq, expected, rtol=0.15, atol=1e-2)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_detects_mean_shift_and_is_translation_invariant(device, dtype):
    torch.manual_seed(0)
    n, d = 512, 3

    x = torch.randn(n, d, device=device, dtype=dtype)
    delta = torch.tensor([1.5, 0.0, -0.75], device=device, dtype=dtype)
    y = torch.randn(n, d, device=device, dtype=dtype) + delta

    v0_rbf = mmd_rbf(x, x, unbiased=False, block_size=256, sigma_seed=0)
    v1_rbf = mmd_rbf(x, y, unbiased=False, block_size=256, sigma_seed=0)

    v0_imq = mmd_imq(x, x, unbiased=False, block_size=256)
    v1_imq = mmd_imq(x, y, unbiased=False, block_size=256)

    assert torch.isfinite(v0_rbf) and torch.isfinite(v1_rbf)
    assert torch.isfinite(v0_imq) and torch.isfinite(v1_imq)

    assert v0_rbf.abs() < 1e-10
    assert v0_imq.abs() < 1e-10
    assert v1_rbf > 1e-4
    assert v1_imq > 1e-4

    shift = torch.tensor([10.0, -3.0, 5.0], device=device, dtype=dtype)
    v1_rbf_shift = mmd_rbf(x + shift, y + shift, unbiased=False, block_size=256, sigma_seed=0)
    v1_imq_shift = mmd_imq(x + shift, y + shift, unbiased=False, block_size=256)

    assert torch.allclose(v1_rbf_shift, v1_rbf, rtol=5e-3, atol=5e-4)
    assert torch.allclose(v1_imq_shift, v1_imq, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_f1_expected_values(device, dtype):
    x_true = torch.tensor([0.0, 1.0, 1.0, 0.0], device=device, dtype=dtype)
    x_gen = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype)

    v = f1(x_true, x_gen, threshold=0.5, zero_division=0.0)
    # x_bin = [0,1,1,0], y_bin=[0,1,0,0] => tp=1, fp=0, fn=1 => F1 = 2/(2+0+1)=2/3
    assert torch.allclose(v, torch.tensor(2.0 / 3.0, device=device, dtype=torch.float32), atol=0.0, rtol=0.0)

    x0 = torch.zeros(10, device=device, dtype=dtype)
    v0 = f1(x0, x0, threshold=0.5, zero_division=0.123)
    assert torch.allclose(v0, torch.tensor(0.123, device=device, dtype=torch.float32), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wasserstein2_1d_shift_exact(device, dtype):
    x = torch.linspace(-2.0, 2.0, 513, device=device, dtype=dtype)
    c = 1.75
    y = x + c

    w2 = wasserstein2_1d(x, y, sqrt=True)
    w2_sq = wasserstein2_1d(x, y, sqrt=False)

    assert torch.allclose(w2, torch.tensor(abs(c), device=device, dtype=dtype), atol=0.0, rtol=0.0)
    assert torch.allclose(w2_sq, torch.tensor(c * c, device=device, dtype=dtype), atol=0.0, rtol=0.0)
