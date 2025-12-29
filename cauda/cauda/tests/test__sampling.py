"""Sampling module unittests."""

# Authors: Hamza Cherkaoui

import pytest
import torch
from cauda._sampling import (sample_scalar_alpha_stable, sample_scaled_scalar_alpha_stable,
                             sample_scaled_isotropic_alpha_stable, sample_spiral, sample_student_t,
                             sample_gaussian, sample_balanced_bimodal_gaussian,
                             sample_unbalanced_bimodal_gaussian)


def _assert_allclose_scalar(x: torch.Tensor, y: float, atol: float, rtol: float = 0.0):
    y_t = torch.tensor(y, device=x.device, dtype=x.dtype)
    assert torch.allclose(x, y_t, atol=atol, rtol=rtol), f"{x.item()} vs {y}"


def _cov(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / float(x.shape[0] - 1)


def test_sample_gaussian_mean_cov_identity_cpu():
    torch.manual_seed(0)
    n, d = 30000, 5
    x = sample_gaussian(n, d, device=torch.device("cpu"), dtype=torch.float64)
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    mean = x.mean(dim=0)
    cov = _cov(x)

    assert mean.abs().max().item() < 0.03

    diag = torch.diag(cov)
    assert (diag - 1.0).abs().max().item() < 0.05

    off = cov - torch.diag(diag)
    assert off.abs().max().item() < 0.03


def test_sample_student_t_mean_and_variance_when_finite():
    torch.manual_seed(1)
    n, d = 60000, 3
    nu = 7.0  # variance exists, Var = nu/(nu-2)
    x = sample_student_t(n, d, nu=nu, device=torch.device("cpu"), dtype=torch.float64)
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    mean = x.mean(dim=0)
    assert mean.abs().max().item() < 0.05

    var_emp = x.var(dim=0, unbiased=True)
    var_true = nu / (nu - 2.0)
    assert (var_emp - var_true).abs().max().item() < 0.12


def test_spiral_radius_moments_no_noise():
    torch.manual_seed(2)
    n = 40000
    turns = 3.0
    R = 4.0
    x = sample_spiral(
        n_samples=n,
        spiral_turns=turns,
        spiral_radius=R,
        spiral_noise=0.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert x.shape == (n, 2)
    assert torch.isfinite(x).all()

    r = torch.linalg.vector_norm(x, ord=2, dim=1)
    assert r.min().item() >= 0.0
    assert r.max().item() <= R + 1e-8

    # If r = R * U with U~Unif(0,1): E[r] = R/2, Var[r] = R^2/12
    _assert_allclose_scalar(r.mean(), R / 2.0, atol=0.03)
    _assert_allclose_scalar(r.var(unbiased=True), (R * R) / 12.0, atol=0.06)


def test_spiral_noise_increases_radius_variance():
    torch.manual_seed(3)
    r_tol = 1e-2
    a_tol = 5e-2
    n = 20000
    R = 4.0
    x0 = sample_spiral(n, spiral_radius=R, spiral_noise=0.0, device="cpu", dtype=torch.float64)
    x1 = sample_spiral(n, spiral_radius=R, spiral_noise=0.25, device="cpu", dtype=torch.float64)

    r0 = torch.linalg.vector_norm(x0, dim=1)
    r1 = torch.linalg.vector_norm(x1, dim=1)

    assert (1 + r_tol) * r1.var(unbiased=True).item() + a_tol > r0.var(unbiased=True).item()


def test_bimodal_balanced_mean_var_and_weight():
    torch.manual_seed(4)
    n, d = 60000, 4
    mu, s = 3.0, 0.2
    x = sample_balanced_bimodal_gaussian(n, d, mu=mu, s=s, device="cpu", dtype=torch.float64)
    assert x.shape == (n, d)
    assert torch.isfinite(x).all()

    mean = x.mean(dim=0)
    assert mean.abs().max().item() < 0.05

    # For p=0.5, Var = s^2 + mu^2 (per coordinate)
    var_emp = x.var(dim=0, unbiased=True)
    var_true = s * s + mu * mu
    assert (var_emp - var_true).abs().max().item() < 0.12

    # With small s, sign estimates mixture weights well (balanced -> ~0.5 negative)
    frac_neg = (x[:, 0] < 0.0).to(torch.float64).mean().item()
    assert abs(frac_neg - 0.5) < 0.02


def test_scalar_alpha_stable_positive_and_scaling_property():
    torch.manual_seed(6)
    n = 60000
    alpha = 0.6
    a1 = sample_scalar_alpha_stable(n, alpha=alpha, scale=1.0, device="cpu", dtype=torch.float64)
    a3 = sample_scalar_alpha_stable(n, alpha=alpha, scale=3.0, device="cpu", dtype=torch.float64)

    assert a1.shape == (n,)
    assert a3.shape == (n,)
    assert (a1 > 0).all()
    assert (a3 > 0).all()
    assert torch.isfinite(a1).all() and torch.isfinite(a3).all()

    q = torch.tensor([0.5, 0.9], dtype=torch.float64)
    q1 = torch.quantile(a1, q)
    q3 = torch.quantile(a3, q)

    ratio = (q3 / q1).cpu()
    # scaling should be ~3 (allow slack: heavy tails -> noisy quantile estimates)
    assert torch.allclose(ratio, torch.full_like(ratio, 3.0), atol=0.0, rtol=0.10)


def test_scaled_scalar_alpha_stable_shape_and_positivity():
    torch.manual_seed(7)
    n = 40000
    alpha = 1.7
    a = sample_scaled_scalar_alpha_stable(n, alpha=alpha, device="cpu", dtype=torch.float64)
    assert a.shape == (n, 1)
    assert torch.isfinite(a).all()
    assert (a > 0).all()


def test_scaled_isotropic_alpha_stable_2d_direction_uniformity_and_symmetry():
    torch.manual_seed(8)
    n = 60000
    alpha = 1.7
    x = sample_scaled_isotropic_alpha_stable(n, dim=2, alpha=alpha, device="cpu", dtype=torch.float64)
    assert x.shape == (n, 2)
    assert torch.isfinite(x).all()

    # Direction should be uniform because X = sqrt(A) * G and G has uniform direction.
    theta = torch.atan2(x[:, 1], x[:, 0])
    c = torch.cos(theta).mean()
    s = torch.sin(theta).mean()
    assert abs(c.item()) < 0.015
    assert abs(s.item()) < 0.015

    # Symmetry: each coordinate should be ~50% positive
    frac_pos0 = (x[:, 0] > 0.0).to(torch.float64).mean().item()
    frac_pos1 = (x[:, 1] > 0.0).to(torch.float64).mean().item()
    assert abs(frac_pos0 - 0.5) < 0.015
    assert abs(frac_pos1 - 0.5) < 0.015


def test_input_validation_alpha_stable():
    with pytest.raises(ValueError):
        _ = sample_scalar_alpha_stable(10, alpha=1.0, scale=1.0, device="cpu")
    with pytest.raises(ValueError):
        _ = sample_scalar_alpha_stable(10, alpha=0.5, scale=0.0, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bimodal_respects_device_cuda():
    # This test is meant to catch device-mismatch bugs (e.g., multinomial probs on CPU).
    torch.manual_seed(0)
    x = sample_balanced_bimodal_gaussian(1024, 2, device=torch.device("cuda"), dtype=torch.float32)
    assert x.device.type == "cuda"
