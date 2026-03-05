"""Neural networks module unittests."""

import pytest
import torch
from cauda.model import TimeEmbedding
from .utils import _devices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _devices())
def test_time_embedding_shape_dtype_device(dtype, device):
    m = TimeEmbedding(hidden=32).to(device=device, dtype=dtype)
    t = torch.rand(17, 1, device=device, dtype=dtype)
    y = m(t)
    assert y.shape == (17, 32)
    assert y.dtype == dtype
    assert y.device.type == device.type
    assert torch.isfinite(y).all()
