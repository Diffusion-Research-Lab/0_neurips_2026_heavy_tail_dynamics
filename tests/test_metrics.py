"""Metrics module unittests."""

import pytest
import torch
from genkit.metrics import fid, mmd_rbf, mssle, sliced_wasserstein, tail_coverage_error
from .utils import _devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fid_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 3, device=device, dtype=dtype)

    score = fid(x, x)

    assert isinstance(score, float)
    assert score == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fid_matches_squared_mean_shift_for_translated_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(4096, 4, device=device, dtype=dtype)
    shift = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    y = x + shift

    score = fid(x, y)
    expected = float((shift * shift).sum().item())

    assert isinstance(score, float)
    assert score == pytest.approx(expected, rel=5e-2, abs=5e-2)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fid_detects_covariance_mismatch(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(4096, 2, device=device, dtype=dtype)
    y = 2.0 * x

    score = fid(x, y)

    assert isinstance(score, float)
    assert score > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mssle_is_zero_for_permuted_identical_empirical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 3, device=device, dtype=dtype)
    y = x[torch.randperm(x.shape[0], device=device)]

    score = mssle(x, y)

    assert isinstance(score, float)
    assert score == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 3, device=device, dtype=dtype)

    sw2 = sliced_wasserstein(x, x, n_projections=64, n_grid=512, seed=123)

    assert isinstance(sw2, float)
    assert sw2 >= 0.0
    assert sw2 == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sliced_wasserstein_translation_formula(device, dtype):
    torch.manual_seed(0)
    n, d = 1024, 4
    x = torch.randn(n, d, device=device, dtype=dtype)
    shift = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    y = x + shift

    sw2 = sliced_wasserstein(x, y, n_projections=256, n_grid=1024, seed=123)
    expected = (shift @ shift).item() / float(d)

    assert isinstance(sw2, float)
    assert sw2 >= 0.0
    assert sw2 == pytest.approx(expected, rel=0.15, abs=1e-2)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_rbf_near_zero_for_identical_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(512, 2, device=device, dtype=dtype)

    score = mmd_rbf(x, x)

    assert isinstance(score, float)
    assert abs(score) < 5e-3


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_rbf_detects_shifted_samples(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(512, 2, device=device, dtype=dtype)
    y = x + 2.0

    score = mmd_rbf(x, y)

    assert isinstance(score, float)
    assert score > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mmd_rbf_supports_biased_and_unbiased_estimators(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(256, 2, device=device, dtype=dtype)
    y = x + 0.5

    score_biased = mmd_rbf(x, y, estimator="biased")
    score_unbiased = mmd_rbf(x, y, estimator="unbiased")

    assert isinstance(score_biased, float)
    assert isinstance(score_unbiased, float)
    assert score_biased >= 0.0
    assert torch.isfinite(torch.tensor(score_unbiased)).item()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_near_zero_for_identical_distributions(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(2048, 3, device=device, dtype=dtype)

    score = tail_coverage_error(x, x)

    assert isinstance(score, float)
    assert score == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_detects_heavier_tailed_generated(device, dtype):
    torch.manual_seed(0)
    x_ref = torch.randn(2048, 2, device=device, dtype=dtype)
    x_gen = 3.0 * torch.randn(2048, 2, device=device, dtype=dtype)

    score = tail_coverage_error(x_ref, x_gen, tail="upper")

    assert isinstance(score, float)
    assert score > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_reduction_none_returns_tensor_per_prob(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, 2, device=device, dtype=dtype)
    probs = [0.9, 0.95, 0.99]

    result = tail_coverage_error(x, x, probs=probs, reduction="none")

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == len(probs)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tce_lower_tail_detects_lighter_left_tail(device, dtype):
    torch.manual_seed(0)
    x_ref = torch.randn(2048, 2, device=device, dtype=dtype)
    x_gen = x_ref + 5.0

    score = tail_coverage_error(x_ref, x_gen, tail="lower")

    assert isinstance(score, float)
    assert score > 0.0
