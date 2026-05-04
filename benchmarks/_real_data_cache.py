"""Preprocessed real-dataset cache helpers for benchmark runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from genkit.datasets import fetch_real_data
from genkit.datasets._dataset import _resolve_real_data_home


CACHE_VERSION = 1
TRUTHY = {"1", "true", "yes", "y", "on"}


def real_data_root(data_root: str | Path | None = None) -> Path:
    """Return the FlowBench real-data root."""
    if data_root is not None:
        return Path(data_root).expanduser()
    return _resolve_real_data_home()


def require_preprocessed_real_data() -> bool:
    """Return whether real benchmark data must be loaded from processed cache."""
    return os.getenv("FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA", "").lower() in TRUTHY


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
    root = real_data_root(data_root)
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
    if name in {"kddcup", "default_credit"}:
        kwargs.setdefault("data_home", str(raw_root / "scikit_learn"))
    elif name in {"earthquakes", "wildfires"}:
        kwargs.setdefault("data_home", str(raw_root / "powerlaws"))
    elif name == "lvis":
        kwargs.setdefault("cache_dir", str(data_root / "processed" / "_image_loader" / name))
    elif name == "cifar100_lt":
        kwargs.setdefault("data_home", str(raw_root / "cifar100_lt"))
        kwargs.setdefault("cache_dir", str(data_root / "processed" / "_image_loader" / name))
    elif name == "imagenet_lt":
        kwargs.setdefault("data_home", str(raw_root / "imagenet_lt"))
        kwargs.setdefault("imagenet_root", "/lustre/fswork/dataset/imagenet")
        kwargs.setdefault("cache_dir", str(data_root / "processed" / "_image_loader" / name))
    return kwargs


def build_preprocessed_real_dataset(
    dataset_cfg: dict[str, Any],
    dtype: torch.dtype,
    *,
    data_root: str | Path | None = None,
    overwrite: bool = False,
    source_config: str | Path | None = None,
) -> dict[str, Any]:
    """Build one final train/val/test real-dataset cache."""
    root = real_data_root(data_root)
    cache_path = real_dataset_cache_path(dataset_cfg, dtype, data_root=root)
    name = str(dataset_cfg["name"])
    if cache_path.is_file() and not overwrite:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
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

    kwargs = _loader_kwargs(dataset_cfg, root)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one preprocessed real train/val/test tensor triplet."""
    cache_path = real_dataset_cache_path(dataset_cfg, dtype, data_root=data_root)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed real dataset cache: {cache_path}. "
            "Run `make dataset` on Jean Zay before launching Slurm runs."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    ok, reason = _validate_cached_payload(payload, dataset_cfg, dtype)
    if not ok:
        raise RuntimeError(f"Preprocessed real dataset cache invalid ({reason}): {cache_path}")
    tensors = tuple(
        payload[key].to(device=device, dtype=dtype) for key in ("x_train", "x_val", "x_test")
    )
    return tensors  # type: ignore[return-value]
