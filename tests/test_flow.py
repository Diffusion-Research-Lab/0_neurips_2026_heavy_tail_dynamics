"""Tests for flow-matching models and samplers."""

import pytest
import torch
from genkit._schedules import flux_shifted_timesteps
from genkit.flow_matching import GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT
from .utils import _devices


class _ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class _OnesNet(torch.nn.Module):
    def forward(self, x, t):
        assert t.shape == (x.size(0), 1)
        return torch.ones_like(x)


def _assert_finite_loss(model_cls, *, t, device, dtype, **kwargs):
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, device=device, fdtype=dtype, **kwargs)
    loss = model.loss(torch.randn(5, 2, dtype=dtype, device=device), t=t)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("t", [3, 0.5])
def test_gaussian_flow_linear_accepts_common_time_formats(t, device, dtype):
    _assert_finite_loss(GaussianFlowLinear, t=t, device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_gaussian_flow_loss_rejects_invalid_t(device, dtype):
    with pytest.raises(ValueError):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, device=device, fdtype=dtype).loss(
            torch.randn(5, 2, dtype=dtype, device=device), t=11
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_other_gaussian_flow_variants_produce_finite_loss(model_cls, kwargs, device, dtype):
    _assert_finite_loss(model_cls, t=0.5, device=device, dtype=dtype, **kwargs)


def test_gaussian_flow_ot_rejects_invalid_sigma_min():
    with pytest.raises(ValueError, match="sigma_min"):
        GaussianFlowOT(net=_ZeroNet(), dim=2, n_steps=10, sigma_min=1.5, device="cpu", fdtype=torch.float32)


@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowLinear, {}), (GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_gaussian_flow_sigma_max_scales_default_source(model_cls, kwargs, monkeypatch):
    def _ones(*size, device=None, dtype=None, **kwargs):
        return torch.ones(*size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _ones)
    model = model_cls(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=2.5, device="cpu", fdtype=torch.float32, **kwargs)

    assert torch.equal(model._sample_source_default(3), torch.full((3, 2), 2.5))


def test_gaussian_flow_rejects_invalid_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10, sigma_max=0.0, device="cpu", fdtype=torch.float32)


@pytest.mark.parametrize("model_cls,kwargs", [(GaussianFlowLinear, {}), (GaussianFlowOT, {"sigma_min": 0.1}), (GaussianFlowDDPM, {})])
def test_gaussian_flow_accepts_image_shaped_batches(model_cls, kwargs):
    model = model_cls(net=_ZeroNet(), dim=(1, 4, 4), n_steps=10, device="cpu", fdtype=torch.float32, **kwargs)
    x = torch.randn(3, 1, 4, 4)

    loss = model.loss(x, t=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.sample(2).shape == (2, 1, 4, 4)


def test_gaussian_flow_check_t_rejects_bool():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(TypeError, match="bool"):
        m._check_t(True, 5)


def test_gaussian_flow_check_t_int_above_n_steps_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError):
        m._check_t(11, 5)


def test_gaussian_flow_check_t_float_tensor_out_of_range_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        m._check_t(torch.tensor([0.5, 1.5, 0.3, 0.4, 0.5]), 5)


def test_gaussian_flow_latent_x1_wrong_dim_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    with pytest.raises(ValueError, match="x1"):
        m._latent(torch.randn(4, 3))  # dim 3, model expects dim 2


def test_gaussian_flow_latent_x0_shape_mismatch_raises():
    m = GaussianFlowLinear(net=_ZeroNet(), dim=2, n_steps=10)
    x1 = torch.randn(4, 2)
    x0 = torch.randn(4, 3)  # wrong dim
    with pytest.raises(ValueError, match="x0"):
        m._latent(x1, x_0=x0)


@pytest.mark.parametrize("sampler", ["euler", "heun", "rk4"])
def test_gaussian_flow_sample_dispatches_named_samplers(sampler):
    base = torch.zeros(2, dtype=torch.float32)
    model = GaussianFlowLinear(
        net=_OnesNet(),
        dim=2,
        n_steps=4,
        t_min=0.0,
        t_max=1.0,
        base_or_sample=base,
        sampler=sampler,
    )

    out, trajectory = model._sample(3, return_trajectory=True)

    assert torch.allclose(out, torch.ones(3, 2))
    assert len(trajectory) == 5


def test_gaussian_flow_uses_constructor_sampler_by_default():
    base = torch.zeros(2, dtype=torch.float32)
    model = GaussianFlowLinear(
        net=_OnesNet(),
        dim=2,
        n_steps=4,
        t_min=0.0,
        t_max=1.0,
        base_or_sample=base,
        sampler="euler",
    )

    out, trajectory = model._sample(3, return_trajectory=True)

    assert torch.allclose(out, torch.ones(3, 2))
    assert len(trajectory) == 5


def test_gaussian_flow_sample_accepts_flux_shifted_schedule():
    base = torch.zeros(2, dtype=torch.float32)
    model = GaussianFlowLinear(
        net=_ZeroNet(),
        dim=2,
        n_steps=4,
        t_min=0.0,
        t_max=1.0,
        base_or_sample=base,
        sampler="heun",
        schedule="flux_shifted",
        image_seq_len=1024,
        sample_steps=32,
    )

    out = model.sample(2)

    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.zeros_like(out))


def test_gaussian_flow_rejects_sampler_and_schedule_aliases():
    with pytest.raises(ValueError, match="adaptive"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, sampler="adaptive")
    with pytest.raises(ValueError, match="flux"):
        GaussianFlowLinear(net=_ZeroNet(), dim=2, schedule="flux", image_seq_len=1024)


def test_flux_shifted_schedule_is_monotone_and_seq_len_dependent():
    small = flux_shifted_timesteps(4, 0.0, 1.0, image_seq_len=256)
    large = flux_shifted_timesteps(4, 0.0, 1.0, image_seq_len=4096)

    assert small[0].item() == 0.0
    assert small[-1].item() == 1.0
    assert torch.all(small[1:] > small[:-1])
    assert torch.all(large[1:] > large[:-1])
    assert large[1].item() > small[1].item()


def test_gaussian_flow_adaptive_heun_smoke():
    base = torch.zeros(2, dtype=torch.float32)
    model = GaussianFlowLinear(
        net=_OnesNet(),
        dim=2,
        n_steps=4,
        t_min=0.0,
        t_max=1.0,
        base_or_sample=base,
        sampler="adaptive_heun",
        h_init=0.25,
    )

    out = model.sample(3)

    assert torch.allclose(out, torch.ones(3, 2))
