"""Public dataset."""

from typing import Any
import pandas as pd
import torch
from ._datasets import ALL_DATASETS, _resolve_dataset, _split_tensorize_frame, coerce_numeric_frame
from .utils import getpop


def fetch_synthetic_data(target_data: str, **kwargs: Any):
    """Fetch train, val, and test tensors from a synthetic dataset."""
    entry = _resolve_dataset(target_data, ALL_DATASETS, dataset_type="synthetic")
    if entry.sampler is None:
        raise RuntimeError(f"Synthetic dataset {target_data!r} has no sampler.")

    n_samples = int(getpop(kwargs, "n_samples", 10_000))
    val_size = float(getpop(kwargs, "val_size", 0.15))
    test_size = float(getpop(kwargs, "test_size", 0.15))
    random_state = int(getpop(kwargs, "random_state", 0))
    standardize = bool(getpop(kwargs, "standardize", False))
    device = getpop(kwargs, "device", "cpu")
    dtype = getpop(kwargs, "dtype", torch.float32)

    sampling_kwargs = {"n_samples": n_samples, "device": "cpu", "dtype": dtype}
    for name, builder in entry.sampler_kwargs_builders.items():
        sampling_kwargs[name] = builder(kwargs)

    samples = entry.sampler(**sampling_kwargs).detach().cpu().numpy()
    frame = pd.DataFrame(samples)
    return _split_tensorize_frame(
        frame,
        split_mode=entry.split_mode,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        standardize=standardize,
        device=device,
        dtype=dtype,
    )


def fetch_real_data(target_data: str, **kwargs: Any):
    """Fetch train, val, and test tensors from a real dataset."""
    entry = _resolve_dataset(target_data, ALL_DATASETS, dataset_type="real")
    if entry.loader is None:
        raise RuntimeError(f"Real dataset {target_data!r} has no loader.")

    val_size = float(kwargs.pop("val_size", 0.15))
    test_size = float(kwargs.pop("test_size", 0.15))
    random_state = int(kwargs.pop("random_state", 0))
    standardize = bool(kwargs.pop("standardize", False))
    device = kwargs.pop("device", "cpu")
    dtype = kwargs.pop("dtype", torch.float32)

    frame = coerce_numeric_frame(entry.loader(**kwargs))
    return _split_tensorize_frame(
        frame,
        split_mode=entry.split_mode,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        standardize=standardize,
        device=device,
        dtype=dtype,
    )


def get_dataset_metadata(target_data: str) -> dict[str, Any]:
    """Return metadata for one synthetic or real dataset."""
    return _resolve_dataset(target_data, ALL_DATASETS).metadata()


def list_datasets() -> list[str]:
    """List available synthetic and real datasets."""
    return sorted(ALL_DATASETS)
