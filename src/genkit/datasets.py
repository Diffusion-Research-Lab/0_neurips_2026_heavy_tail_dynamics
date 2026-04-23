"""Public dataset."""

import warnings
from typing import Any
import pandas as pd
import torch
from ._datasets import ALL_DATASETS, _resolve_dataset, _split_tensorize_frame, coerce_numeric_frame, split_tensor_data
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
    """Fetch a real dataset as split tensors."""
    entry = _resolve_dataset(target_data, ALL_DATASETS, dataset_type="real")
    if entry.loader is None:
        raise RuntimeError(f"Real dataset {target_data!r} has no loader.")

    n_samples = kwargs.pop("n_samples", None)
    if n_samples is not None:
        n_samples = int(n_samples)
    val_size = float(kwargs.pop("val_size", 0.15))
    test_size = float(kwargs.pop("test_size", 0.15))
    random_state = int(kwargs.pop("random_state", 0))
    _standardize_sentinel = kwargs.pop("standardize", None)
    standardize = bool(entry.standardize_default if _standardize_sentinel is None else _standardize_sentinel)
    device = kwargs.pop("device", "cpu")
    dtype = kwargs.pop("dtype", torch.float32)

    loaded = entry.loader(**kwargs)
    if isinstance(loaded, pd.DataFrame):
        frame = coerce_numeric_frame(loaded)
        if n_samples is not None:
            if n_samples > len(frame):
                warnings.warn(
                    f"n_samples={n_samples} exceeds available samples ({len(frame)}); returning all.",
                    stacklevel=2,
                )
            frame = frame.sample(n=min(n_samples, len(frame)), random_state=random_state).reset_index(drop=True)
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

    data = torch.as_tensor(loaded)
    if n_samples is not None:
        if n_samples > data.shape[0]:
            warnings.warn(
                f"n_samples={n_samples} exceeds available samples ({data.shape[0]}); returning all.",
                stacklevel=2,
            )
        rng = torch.Generator().manual_seed(random_state)
        idx = torch.randperm(data.shape[0], generator=rng)[: min(n_samples, data.shape[0])]
        data = data[idx]
    x_train, x_val, x_test = split_tensor_data(
        data,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=entry.split_mode,
    )
    x_train = x_train.to(dtype=dtype)
    x_val = x_val.to(dtype=dtype)
    x_test = x_test.to(dtype=dtype)
    if standardize:
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
    return (
        x_train.to(device=device, dtype=dtype),
        x_val.to(device=device, dtype=dtype),
        x_test.to(device=device, dtype=dtype),
    )


def get_dataset_metadata(target_data: str) -> dict[str, Any]:
    """Return metadata for one synthetic or real dataset."""
    return _resolve_dataset(target_data, ALL_DATASETS).metadata()


def list_datasets() -> list[str]:
    """List available synthetic and real datasets."""
    return sorted(ALL_DATASETS)
