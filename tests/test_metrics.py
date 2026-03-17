"""Metrics module unittests."""

import math
import pytest
import torch
from genkit.metrics import sliced_wasserstein2, msle, msle_90, msle_99, mse, rnmse, mae
from genkit.inspect import nn_dist_min
from .utils import _devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mse_mae_basic(device, dtype):
    x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    y = torch.tensor([2.0, 2.0, 1.0], device=device, dtype=dtype)

    assert torch.allclose(mse(x, y), torch.tensor((1.0 + 0.0 + 4.0) / 3.0, device=device, dtype=dtype))
    assert torch.allclose(mae(x, y), torch.tensor((1.0 + 0.0 + 2.0) / 3.0, device=device, dtype=dtype))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rnmse_scale_invariant(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(256, 4, device=device, dtype=dtype)
    y = x + 0.1
    a = 7.0

    v1 = rnmse(x, y)
    v2 = rnmse(a * x, a * y)
    assert torch.allclose(v1, v2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_msle_quantile_variants(device, dtype):
    torch.manual_seed(0)
    x = torch.exp(torch.randn(4096, device=device, dtype=dtype))
    c = 3.5
    y = c * x

    v = msle(x, y)
    v90 = msle_90(x, y)
    v99 = msle_99(x, y)

    assert v.shape == ()
    assert v90.shape == ()
    assert v99.shape == ()
    assert torch.isfinite(v) and torch.isfinite(v90) and torch.isfinite(v99)
    assert v90 >= 0 and v99 >= 0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein2_translation_formula(device, dtype):
    torch.manual_seed(0)
    n, d = 512, 4
    n_proj = 256

    x = torch.randn(n, d, device=device, dtype=dtype)
    c = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    y = x + c

    sw2_sq = sliced_wasserstein2(x, y, n_projections=n_proj, sqrt=False, seed=123)
    expected = (c @ c) / float(d)

    assert sw2_sq.shape == ()
    assert torch.isfinite(sw2_sq)
    assert sw2_sq >= 0
    assert torch.allclose(sw2_sq, expected, rtol=0.15, atol=1e-2)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nn_dist_min_l1_exact_match(device, dtype):
    x_train = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], device=device, dtype=dtype)
    x_gen = torch.tensor([[0.9, 0.1], [0.1, 0.9], [-0.9, 0.05]], device=device, dtype=dtype)

    dmin = nn_dist_min(x_gen, x_train, metric="l1", block_size=2)
    assert math.isfinite(dmin)
    assert dmin >= 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nn_dist_min_l2(device, dtype):
    torch.manual_seed(0)
    x_train = torch.randn(256, 8, device=device, dtype=dtype)
    x_gen = x_train + 0.01 * torch.randn(256, 8, device=device, dtype=dtype)

    dmin = nn_dist_min(x_gen, x_train, metric="sd", block_size=64)
    assert math.isfinite(dmin)
    assert dmin >= 0.0
