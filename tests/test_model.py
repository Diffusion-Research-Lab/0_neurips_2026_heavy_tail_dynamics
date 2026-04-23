"""Neural networks module unittests."""

import pytest
import torch
from genkit.nn import MLPModel, timestep_embedding
from .utils import _devices


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
