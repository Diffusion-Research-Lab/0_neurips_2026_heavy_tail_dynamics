"""Public dataset."""

from pathlib import Path
import warnings
from typing import Any
import pandas as pd
import torch
from ._datasets import (
    ALL_DATASETS,
    DatasetPayload,
    _resolve_dataset,
    _standardize_split_arrays,
    coerce_numeric_frame,
    split_sample_indices,
    to_tensor_triplet,
)
from .utils import getpop


def _metadata_value(value: Any) -> Any:
    """Convert common runtime objects to metadata-friendly values."""
    if isinstance(value, dict):
        return {str(key): _metadata_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    return value


def _split_frame_to_tensors(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[Any, Any, Any]]:
    """Split a frame into tensors and return the exact row indices used."""
    train_idx, val_idx, test_idx = split_sample_indices(
        len(frame),
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    x_train = frame.iloc[train_idx].reset_index(drop=True).to_numpy()
    x_val = frame.iloc[val_idx].reset_index(drop=True).to_numpy()
    x_test = frame.iloc[test_idx].reset_index(drop=True).to_numpy()
    if standardize:
        x_train, x_val, x_test = _standardize_split_arrays(x_train, x_val, x_test)
    return (
        to_tensor_triplet(x_train, x_val, x_test, device=device, dtype=dtype),
        (train_idx, val_idx, test_idx),
    )


def _split_tensor_to_tensors(
    data: torch.Tensor,
    *,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[Any, Any, Any]]:
    """Split a tensor along the sample axis and return the exact row indices used."""
    train_idx, val_idx, test_idx = split_sample_indices(
        int(data.shape[0]),
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )

    def index_rows(indices) -> torch.Tensor:
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        return data.index_select(0, index_tensor)

    x_train = index_rows(train_idx).to(dtype=dtype)
    x_val = index_rows(val_idx).to(dtype=dtype)
    x_test = index_rows(test_idx).to(dtype=dtype)
    if standardize:
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
    return (
        (
            x_train.to(device=device, dtype=dtype),
            x_val.to(device=device, dtype=dtype),
            x_test.to(device=device, dtype=dtype),
        ),
        (train_idx, val_idx, test_idx),
    )


def _split_records(records: list[dict[str, Any]], indices) -> list[dict[str, Any]]:
    """Select records matching split indices."""
    return [records[int(index)] for index in indices]


def _record_histograms(records: list[dict[str, Any]]) -> dict[str, dict[Any, int]]:
    """Build image-level category and frequency histograms from dataset-specific records."""
    category_histogram: dict[int, int] = {}
    frequency_histogram: dict[str, int] = {}
    for record in records:
        for category_id in set(int(value) for value in record.get("category_ids", [])):
            category_histogram[category_id] = category_histogram.get(category_id, 0) + 1
        for frequency in set(str(value) for value in record.get("category_frequencies", []) if value):
            frequency_histogram[frequency] = frequency_histogram.get(frequency, 0) + 1
    return {
        "category_histogram": category_histogram,
        "frequency_histogram": frequency_histogram,
    }


def _split_metadata(indices, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build metadata for one returned split."""
    split_meta: dict[str, Any] = {
        "indices": [int(index) for index in indices],
        "n_samples": int(len(indices)),
    }
    if records is not None:
        split_records = _split_records(records, indices)
        split_meta["records"] = split_records
        split_meta.update(_record_histograms(split_records))
    return split_meta


def _build_return_metadata(
    *,
    entry,
    kind: str,
    target_data: str,
    params: dict[str, Any],
    split_config: dict[str, Any],
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
    split_indices: tuple[Any, Any, Any],
    payload_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble dataset-level and split-level metadata for optional API returns."""
    payload_metadata = dict(payload_metadata or {})
    records = payload_metadata.pop("records", None)
    train_idx, val_idx, test_idx = split_indices
    return {
        "dataset": entry.metadata(),
        "request": {
            "kind": kind,
            "name": target_data,
            "params": _metadata_value(params),
            "split": _metadata_value(split_config),
            "standardize": bool(standardize),
            "device": str(device),
            "dtype": str(dtype),
        },
        "loader": _metadata_value(payload_metadata),
        "splits": {
            "train": _split_metadata(train_idx, records),
            "val": _split_metadata(val_idx, records),
            "test": _split_metadata(test_idx, records),
        },
    }


def fetch_synthetic_data(target_data: str, **kwargs: Any):
    """Fetch train, val, and test tensors from a synthetic dataset."""
    return_metadata = bool(getpop(kwargs, "return_metadata", False))
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
    tensors, split_indices = _split_frame_to_tensors(
        frame,
        split_mode=entry.split_mode,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        standardize=standardize,
        device=device,
        dtype=dtype,
    )
    if not return_metadata:
        return tensors
    metadata = _build_return_metadata(
        entry=entry,
        kind="synthetic",
        target_data=target_data,
        params=sampling_kwargs,
        split_config={"val_size": val_size, "test_size": test_size, "random_state": random_state},
        standardize=standardize,
        device=device,
        dtype=dtype,
        split_indices=split_indices,
    )
    return (*tensors, metadata)


def fetch_real_data(target_data: str, **kwargs: Any):
    """Fetch a real dataset as split tensors."""
    return_metadata = bool(kwargs.pop("return_metadata", False))
    entry = _resolve_dataset(target_data, ALL_DATASETS, dataset_type="real")
    if entry.loader is None:
        raise RuntimeError(f"Real dataset {target_data!r} has no loader.")

    n_samples = kwargs.pop("n_samples", None)
    if n_samples is not None:
        n_samples = int(n_samples)
    if getattr(entry, "name", target_data) == "lvis" and n_samples is not None and "max_samples" not in kwargs:
        kwargs["max_samples"] = n_samples
        n_samples = None
    val_size = float(kwargs.pop("val_size", 0.15))
    test_size = float(kwargs.pop("test_size", 0.15))
    random_state = int(kwargs.pop("random_state", 0))
    _standardize_sentinel = kwargs.pop("standardize", None)
    standardize = bool(entry.standardize_default if _standardize_sentinel is None else _standardize_sentinel)
    device = kwargs.pop("device", "cpu")
    dtype = kwargs.pop("dtype", torch.float32)

    loader_kwargs = dict(kwargs)
    if n_samples is not None:
        loader_kwargs["n_samples"] = n_samples
    loaded = entry.loader(**kwargs)
    payload_metadata: dict[str, Any] = {}
    if isinstance(loaded, DatasetPayload):
        payload_metadata = dict(loaded.metadata)
        loaded = loaded.data

    if isinstance(loaded, pd.DataFrame):
        frame = coerce_numeric_frame(loaded)
        records = payload_metadata.get("records")
        if n_samples is not None:
            if n_samples > len(frame):
                warnings.warn(
                    f"n_samples={n_samples} exceeds available samples ({len(frame)}); returning all.",
                    stacklevel=2,
                )
            selected = frame.sample(n=min(n_samples, len(frame)), random_state=random_state).index.to_numpy()
            frame = frame.loc[selected].reset_index(drop=True)
            if records is not None:
                payload_metadata["records"] = _split_records(records, selected)
        tensors, split_indices = _split_frame_to_tensors(
            frame,
            split_mode=entry.split_mode,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            standardize=standardize,
            device=device,
            dtype=dtype,
        )
        if not return_metadata:
            return tensors
        metadata = _build_return_metadata(
            entry=entry,
            kind="real",
            target_data=target_data,
            params=loader_kwargs,
            split_config={"val_size": val_size, "test_size": test_size, "random_state": random_state},
            standardize=standardize,
            device=device,
            dtype=dtype,
            split_indices=split_indices,
            payload_metadata=payload_metadata,
        )
        return (*tensors, metadata)

    data = torch.as_tensor(loaded)
    records = payload_metadata.get("records")
    if n_samples is not None:
        if n_samples > data.shape[0]:
            warnings.warn(
                f"n_samples={n_samples} exceeds available samples ({data.shape[0]}); returning all.",
                stacklevel=2,
            )
        rng = torch.Generator().manual_seed(random_state)
        idx = torch.randperm(data.shape[0], generator=rng)[: min(n_samples, data.shape[0])]
        data = data[idx]
        if records is not None:
            payload_metadata["records"] = _split_records(records, idx.tolist())
    tensors, split_indices = _split_tensor_to_tensors(
        data,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=entry.split_mode,
        standardize=standardize,
        device=device,
        dtype=dtype,
    )
    if not return_metadata:
        return tensors
    metadata = _build_return_metadata(
        entry=entry,
        kind="real",
        target_data=target_data,
        params=loader_kwargs,
        split_config={"val_size": val_size, "test_size": test_size, "random_state": random_state},
        standardize=standardize,
        device=device,
        dtype=dtype,
        split_indices=split_indices,
        payload_metadata=payload_metadata,
    )
    return (*tensors, metadata)


def get_dataset_metadata(target_data: str) -> dict[str, Any]:
    """Return metadata for one synthetic or real dataset."""
    return _resolve_dataset(target_data, ALL_DATASETS).metadata()


def list_datasets() -> list[str]:
    """List available synthetic and real datasets."""
    return sorted(ALL_DATASETS)
