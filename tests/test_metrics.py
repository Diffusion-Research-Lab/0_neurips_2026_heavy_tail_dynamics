"""Metrics module unittests."""

import pytest
import torch

from genkit.metrics import mssle_90, mssle_95, sliced_wasserstein2, wasserstein_distance
from .utils import _devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mssle_tail_variants_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(2048, 2, device=device, dtype=dtype)

    v90 = mssle_90(x, x)
    v95 = mssle_95(x, x)

    assert isinstance(v90, float)
    assert isinstance(v95, float)
    assert v90 >= 0.0 and v95 >= 0.0
    assert v90 == pytest.approx(0.0, abs=1e-12)
    assert v95 == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mssle_tail_detects_tail_rescaling(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(4096, 2, device=device, dtype=dtype)
    y = 3.0 * x

    v90 = mssle_90(x, y)
    v95 = mssle_95(x, y)

    assert torch.isfinite(torch.tensor(v90))
    assert torch.isfinite(torch.tensor(v95))
    assert v90 > 0.0
    assert v95 > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wasserstein_distance_1d_translation_formula(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(4096, 1, device=device, dtype=dtype)
    shift = torch.tensor(2.5, device=device, dtype=dtype)
    y = x + shift

    w1 = wasserstein_distance(x, y, p=1, n_grid=2048)
    w2 = wasserstein_distance(x, y, p=2, n_grid=2048)

    assert isinstance(w1, float)
    assert isinstance(w2, float)
    assert w1 == pytest.approx(shift.item(), rel=5e-2, abs=5e-2)
    assert w2 == pytest.approx(shift.item(), rel=5e-2, abs=5e-2)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein2_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 3, device=device, dtype=dtype)

    sw2 = sliced_wasserstein2(x, x, n_projections=64, n_grid=512, seed=123)

    assert isinstance(sw2, float)
    assert sw2 >= 0.0
    assert sw2 == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein2_translation_formula(device, dtype):
    torch.manual_seed(0)
    n, d = 1024, 4
    x = torch.randn(n, d, device=device, dtype=dtype)
    shift = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    y = x + shift

    sw2 = sliced_wasserstein2(x, y, n_projections=256, n_grid=1024, seed=123)
    expected = (shift @ shift).item() / float(d)

    assert isinstance(sw2, float)
    assert sw2 >= 0.0
    assert sw2 == pytest.approx(expected, rel=0.15, abs=1e-2)
