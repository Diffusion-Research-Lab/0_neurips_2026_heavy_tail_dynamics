"""Preprocessed real-dataset cache helpers for benchmark runs."""

import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import torch
from jeanzaydata._dataset import (
    REAL_DATASETS,
    _build_return_metadata,
    _print_image_loader_progress,
    _resolve_real_data_home,
    fetch_real_data,
    read_rgb_resized,
    split_sample_indices,
)
from jeanzaydata._imagenet_lt import (
    _imagenet_lt_record,
    _normalize_imagenet_lt_split,
    _parse_imagenet_lt_split_file,
    _resolve_imagenet_lt_annotation,
    _resolve_imagenet_lt_image_path,
    _resolve_imagenet_root,
    _select_imagenet_lt_records,
)

CACHE_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    return value


def real_dataset_request(dataset_cfg: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    """Return the canonical request that defines one processed real cache."""
    if str(dataset_cfg.get("kind", "")).lower() != "real":
        raise ValueError("Only real datasets can use the preprocessed real-data cache.")
    name = str(dataset_cfg.get("name", "")).strip()
    if not name:
        raise ValueError("dataset.name must be provided.")
    return {
        "cache_version": CACHE_VERSION,
        "kind": "real",
        "name": name,
        "params": _jsonable(copy.deepcopy(dataset_cfg.get("params", {}))),
        "split": _jsonable(copy.deepcopy(dataset_cfg.get("split", {}))),
        "dtype": str(dtype).replace("torch.", ""),
    }


def real_dataset_cache_key(dataset_cfg: dict[str, Any], dtype: torch.dtype) -> str:
    """Return a stable short hash for one real-dataset request."""
    payload = json.dumps(real_dataset_request(dataset_cfg, dtype), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def real_dataset_cache_path(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Return the processed tensor cache path for one real-dataset request."""
    root = Path(data_root).expanduser() if data_root is not None else _resolve_real_data_home()
    name = str(dataset_cfg["name"]).strip()
    key = real_dataset_cache_key(dataset_cfg, dtype)
    return root / "processed" / name / f"{name}_{key}.pt"


def _validate_cached_payload(
    payload: Any,
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
) -> tuple[bool, str]:
    """Return (ok, reason) for whether a loaded cache payload is reusable."""
    if not isinstance(payload, dict):
        return False, "payload is not a dict"
    if payload.get("cache_version") != CACHE_VERSION:
        return False, (
            f"cache_version mismatch (got {payload.get('cache_version')!r}, "
            f"expected {CACHE_VERSION})"
        )
    expected_key = real_dataset_cache_key(dataset_cfg, dtype)
    if payload.get("cache_key") != expected_key:
        return False, "cache_key mismatch (config or dtype changed since the cache was built)"
    expected_name = str(dataset_cfg.get("name", "")).strip()
    cached_name = str((payload.get("request") or {}).get("name", "")).strip()
    if expected_name and cached_name and cached_name != expected_name:
        return False, f"dataset name mismatch (cache={cached_name!r}, config={expected_name!r})"
    for key in ("x_train", "x_val", "x_test"):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            return False, f"missing or invalid tensor {key!r}"
    train_n = int(payload["x_train"].shape[0])
    val_n = int(payload["x_val"].shape[0])
    test_n = int(payload["x_test"].shape[0])
    if min(train_n, val_n, test_n) <= 0:
        return False, f"empty split detected (train={train_n}, val={val_n}, test={test_n})"
    sample_shape = tuple(payload["x_train"].shape[1:])
    for key in ("x_val", "x_test"):
        other_shape = tuple(payload[key].shape[1:])
        if other_shape != sample_shape:
            return False, (
                f"split sample shape mismatch (x_train{sample_shape} vs {key}{other_shape})"
            )
    params = dataset_cfg.get("params", {}) or {}
    requested = params.get("max_samples", params.get("n_samples"))
    total_n = train_n + val_n + test_n
    if requested is not None:
        try:
            requested_int = int(requested)
        except (TypeError, ValueError):
            requested_int = None
        if requested_int is not None and requested_int > 0 and total_n > requested_int:
            return False, (
                f"cached sample count {total_n} exceeds requested {requested_int} "
                "(stale cache from a larger config)"
            )
    return True, "ok"


def _loader_kwargs(dataset_cfg: dict[str, Any], data_root: Path) -> dict[str, Any]:
    kwargs = {**copy.deepcopy(dataset_cfg.get("params", {})), **copy.deepcopy(dataset_cfg.get("split", {}))}
    name = str(dataset_cfg["name"]).strip()
    raw_root = data_root / "raw"
    if name == "lvis":
        kwargs.setdefault("cache_dir", str(data_root / "processed" / "_image_loader" / name))
    elif name == "cifar100_lt":
        kwargs.setdefault("data_home", str(raw_root / "cifar100_lt"))
        kwargs.setdefault("cache_dir", str(data_root / "processed" / "_image_loader" / name))
    elif name == "imagenet_lt":
        kwargs.setdefault("data_home", str(raw_root / "imagenet_lt"))
        kwargs.setdefault("imagenet_root", "/lustre/fswork/dataset/imagenet")
        kwargs.setdefault("cache_dir", str(data_root / "processed" / "_image_loader" / name))
    return kwargs


def _build_imagenet_lt_split_cache(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    root: Path,
    cache_path: Path,
    source_config: str | Path | None,
) -> dict[str, Any] | None:
    """Build ImageNet-LT split tensors directly to avoid a full unsplit tensor."""
    kwargs = _loader_kwargs(dataset_cfg, root)
    n_samples = kwargs.pop("n_samples", None)
    if n_samples is not None and "max_samples" not in kwargs:
        kwargs["max_samples"] = n_samples

    val_size = float(kwargs.pop("val_size", 0.15))
    test_size = float(kwargs.pop("test_size", 0.15))
    random_state = int(kwargs.pop("random_state", 0))
    standardize_value = kwargs.pop("standardize", None)
    standardize = bool(REAL_DATASETS["imagenet_lt"].standardize_default if standardize_value is None else standardize_value)
    if standardize:
        return None

    split = _normalize_imagenet_lt_split(str(kwargs.pop("split", "train")))
    image_size = int(kwargs.pop("image_size", 64))
    max_samples_raw = kwargs.pop("max_samples", None)
    max_samples = None if max_samples_raw is None else int(max_samples_raw)
    seed = int(kwargs.pop("seed", 0))
    data_home = kwargs.pop("data_home", None)
    annotation_path_arg = kwargs.pop("annotation_path", None)
    imagenet_root_arg = kwargs.pop("imagenet_root", None)
    kwargs.pop("cache", None)
    kwargs.pop("cache_dir", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected ImageNet-LT loader kwargs: {unexpected}.")
    if image_size <= 0:
        raise ValueError("image_size must be > 0.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    annotation_path = _resolve_imagenet_lt_annotation(
        data_home=data_home,
        split=split,
        annotation_path=annotation_path_arg,
    )
    imagenet_root = _resolve_imagenet_root(data_home=data_home, imagenet_root=imagenet_root_arg)
    raw_records = _parse_imagenet_lt_split_file(annotation_path)
    raw_records = _select_imagenet_lt_records(raw_records, max_samples=max_samples, seed=seed)
    if not raw_records:
        raise RuntimeError(f"ImageNet-LT produced no records for split={split!r} in {annotation_path}.")

    total_records = len(raw_records)
    train_idx, val_idx, test_idx = split_sample_indices(
        total_records,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=REAL_DATASETS["imagenet_lt"].split_mode,
    )
    split_indices = (train_idx, val_idx, test_idx)
    split_tensors = {
        "train": torch.empty((len(train_idx), 3, image_size, image_size), dtype=dtype),
        "val": torch.empty((len(val_idx), 3, image_size, image_size), dtype=dtype),
        "test": torch.empty((len(test_idx), 3, image_size, image_size), dtype=dtype),
    }
    destinations: dict[int, tuple[str, int]] = {}
    for split_name, indices in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        for position, source_index in enumerate(indices):
            destinations[int(source_index)] = (split_name, int(position))

    records = []
    histogram: dict[int, int] = {}
    started_at = time.perf_counter()
    print(
        f"[dataset] image imagenet_lt  start total={total_records} split={split} "
        f"image_size={image_size} root={imagenet_root}",
        flush=True,
    )
    for index, raw_record in enumerate(raw_records):
        image_path = _resolve_imagenet_lt_image_path(raw_record, imagenet_root)
        split_name, position = destinations[index]
        split_tensors[split_name][position].copy_(read_rgb_resized(image_path, image_size).to(dtype=dtype))
        record = _imagenet_lt_record(raw_record, image_path)
        records.append(record)
        class_id = int(record["class_id"])
        histogram[class_id] = histogram.get(class_id, 0) + 1
        current = index + 1
        if current == total_records or current == 1 or current % 1000 == 0:
            _print_image_loader_progress("imagenet_lt", current, total_records, started_at)

    loader_params = {
        "split": split,
        "image_size": image_size,
        "max_samples": max_samples,
        "seed": seed,
        "data_home": data_home,
        "annotation_path": annotation_path_arg,
        "imagenet_root": imagenet_root_arg,
    }
    loader_params = {key: value for key, value in loader_params.items() if value is not None}
    metadata = _build_return_metadata(
        entry=REAL_DATASETS["imagenet_lt"],
        kind="real",
        target_data="imagenet_lt",
        params=loader_params,
        split_config={"val_size": val_size, "test_size": test_size, "random_state": random_state},
        standardize=standardize,
        device="cpu",
        dtype=dtype,
        split_indices=split_indices,
        payload_metadata={
            "source_split": split,
            "annotation_path": str(annotation_path),
            "imagenet_root": str(imagenet_root),
            "image_size": int(image_size),
            "n_selected_images": total_records,
            "labels": [int(record["class_id"]) for record in records],
            "class_histogram": histogram,
            "records": records,
            "cache_hit": False,
        },
    )
    request = real_dataset_request(dataset_cfg, dtype)
    payload = {
        "cache_version": CACHE_VERSION,
        "cache_key": real_dataset_cache_key(dataset_cfg, dtype),
        "request": request,
        "source_config": None if source_config is None else str(source_config),
        "created_at_ns": time.time_ns(),
        "x_train": split_tensors["train"].contiguous(),
        "x_val": split_tensors["val"].contiguous(),
        "x_test": split_tensors["test"].contiguous(),
        "metadata": metadata,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    return {
        "status": "built",
        "path": str(cache_path),
        "dataset": str(dataset_cfg["name"]),
        "train_shape": tuple(split_tensors["train"].shape),
        "val_shape": tuple(split_tensors["val"].shape),
        "test_shape": tuple(split_tensors["test"].shape),
    }


def build_preprocessed_real_dataset(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    data_root: str | Path | None = None,
    overwrite: bool = False,
    source_config: str | Path | None = None,
) -> dict[str, Any]:
    """Build one final train/val/test real-dataset cache."""
    root = Path(data_root).expanduser() if data_root is not None else _resolve_real_data_home()
    cache_path = real_dataset_cache_path(dataset_cfg, dtype, data_root=root)
    name = str(dataset_cfg["name"])
    if cache_path.is_file() and not overwrite:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
        except Exception as exc:
            print(
                f"[REBUILD] {name}: torch.load failed ({type(exc).__name__}: {exc}) -> {cache_path}",
                flush=True,
            )
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass
        else:
            ok, reason = _validate_cached_payload(payload, dataset_cfg, dtype)
            if ok:
                x_train = payload["x_train"]
                x_val = payload["x_val"]
                x_test = payload["x_test"]
                print(f"[SKIP] Valid preprocessed cache found for {name}: {cache_path}", flush=True)
                return {
                    "status": "exists",
                    "path": str(cache_path),
                    "dataset": name,
                    "train_shape": tuple(x_train.shape),
                    "val_shape": tuple(x_val.shape),
                    "test_shape": tuple(x_test.shape),
                }
            print(f"[REBUILD] {name}: {reason} -> {cache_path}", flush=True)
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass

    if name == "imagenet_lt":
        result = _build_imagenet_lt_split_cache(
            dataset_cfg,
            dtype,
            root=root,
            cache_path=cache_path,
            source_config=source_config,
        )
        if result is not None:
            return result

    kwargs = _loader_kwargs(dataset_cfg, root)
    if name in {"lvis", "cifar100_lt", "imagenet_lt"}:
        kwargs.setdefault("cache", False)
    x_train, x_val, x_test, metadata = fetch_real_data(
        str(dataset_cfg["name"]),
        **kwargs,
        dtype=dtype,
        device="cpu",
        return_metadata=True,
    )
    request = real_dataset_request(dataset_cfg, dtype)
    payload = {
        "cache_version": CACHE_VERSION,
        "cache_key": real_dataset_cache_key(dataset_cfg, dtype),
        "request": request,
        "source_config": None if source_config is None else str(source_config),
        "created_at_ns": time.time_ns(),
        "x_train": x_train.detach().cpu().contiguous(),
        "x_val": x_val.detach().cpu().contiguous(),
        "x_test": x_test.detach().cpu().contiguous(),
        "metadata": metadata,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    return {
        "status": "built",
        "path": str(cache_path),
        "dataset": str(dataset_cfg["name"]),
        "train_shape": tuple(x_train.shape),
        "val_shape": tuple(x_val.shape),
        "test_shape": tuple(x_test.shape),
    }


def load_preprocessed_real_dataset(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    device: str | torch.device = "cpu",
    data_root: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    mmap: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Load selected preprocessed real dataset splits."""
    cache_path = real_dataset_cache_path(dataset_cfg, dtype, data_root=data_root)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed real dataset cache: {cache_path}. "
            "Run `make dataset` on Jean Zay before launching Slurm runs."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=mmap)
    ok, reason = _validate_cached_payload(payload, dataset_cfg, dtype)
    if not ok:
        raise RuntimeError(f"Preprocessed real dataset cache invalid ({reason}): {cache_path}")
    split_keys = {"train": "x_train", "val": "x_val", "test": "x_test"}
    unknown = sorted(set(splits) - set(split_keys))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}. Expected any of {sorted(split_keys)}.")
    tensors = tuple(
        payload[split_keys[split]].to(device=device, dtype=dtype) for split in splits
    )
    return tensors  # type: ignore[return-value]


def load_preprocessed_real_dataset_shapes(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    data_root: str | Path | None = None,
    mmap: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Return cached split shapes without copying tensor payloads into RAM."""
    cache_path = real_dataset_cache_path(dataset_cfg, dtype, data_root=data_root)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed real dataset cache: {cache_path}. "
            "Run `make dataset` on Jean Zay before launching Slurm runs."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=mmap)
    ok, reason = _validate_cached_payload(payload, dataset_cfg, dtype)
    if not ok:
        raise RuntimeError(f"Preprocessed real dataset cache invalid ({reason}): {cache_path}")
    return {
        "train": tuple(payload["x_train"].shape),
        "val": tuple(payload["x_val"].shape),
        "test": tuple(payload["x_test"].shape),
    }


def load_preprocessed_real_dataset_metadata(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load metadata stored next to one preprocessed real-dataset tensor cache."""
    cache_path = real_dataset_cache_path(dataset_cfg, dtype, data_root=data_root)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed real dataset cache: {cache_path}. "
            "Run `make dataset` on Jean Zay before launching Slurm runs."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    ok, reason = _validate_cached_payload(payload, dataset_cfg, dtype)
    if not ok:
        raise RuntimeError(f"Preprocessed real dataset cache invalid ({reason}): {cache_path}")
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}
