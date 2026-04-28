"""Private dataset helpers and registries."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Callable
from urllib.request import urlretrieve
import warnings
import numpy as np
import requests
from tqdm import tqdm
import pandas as pd
import torch
import xarray as xr
from sklearn.datasets import fetch_kddcup99, fetch_openml
from sklearn.model_selection import train_test_split
from .utils import getpop
from ._sampling import (
    sample_checker,
    sample_exponential,
    sample_gaussian,
    sample_scaled_isotropic_alpha_stable,
    sample_spiral,
    sample_student_t,
    sample_unbalanced_highdim_alpha_stable_mixture,
    sample_unbalanced_highdim_gaussian_mixture,
)


@dataclass(frozen=True)
class DatasetPayload:
    """Container for loaded data plus optional per-sample metadata."""

    data: pd.DataFrame | torch.Tensor | np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


DatasetLoader = Callable[..., pd.DataFrame | torch.Tensor | np.ndarray | DatasetPayload]
DatasetSampler = Callable[..., torch.Tensor]
SamplerKwargBuilders = dict[str, Callable[[dict[str, Any]], Any]]


@dataclass(frozen=True)
class DatasetEntry:
    """Describe one dataset exposed through the public dataset API."""

    name: str
    dataset_type: str
    description: str
    tail_index_alpha: Any = None
    split_mode: str = "random"
    standardize_default: bool = True
    dim: int | tuple[int, ...] | None = None
    n_samples: int | None = None
    loader: DatasetLoader | None = None
    sampler: DatasetSampler | None = None
    sampler_kwargs_builders: SamplerKwargBuilders = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Return the public metadata view of the dataset."""
        return {
            "name": self.name,
            "tail_index_alpha": self.tail_index_alpha,
            "description": self.description,
            "split_mode": self.split_mode,
            "dataset_type": self.dataset_type,
            "dim": self.dim,
            "n_samples": self.n_samples,
        }


def _copy_if_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Copy NumPy arrays before tensor conversion to avoid non-writable views."""
    if isinstance(array, np.ndarray):
        return np.array(array, copy=True)
    return array


def _decode_byte_string(value: Any) -> Any:
    """Decode byte-string values to UTF-8 strings."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _resolve_real_data_home() -> Path:
    """Pick the default persistent cache root for real datasets."""
    candidates: list[Path] = []
    env_root = os.getenv("FLOWBENCH_DATA_HOME")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    work_root = os.getenv("WORK")
    if work_root:
        candidates.append(Path(work_root).expanduser() / "flowbench_data")
    home_root = os.getenv("HOME")
    if home_root:
        candidates.append(Path(home_root).expanduser() / ".cache" / "flowbench_data")
    candidates.append(Path("/tmp") / "flowbench_data")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate

    raise RuntimeError("Unable to create a real-dataset cache directory.")


def _cache_remote_text_file(url: str, *, data_home: str | Path, filename: str) -> Path:
    """Download one raw text dataset once and reuse it locally."""
    cache_dir = Path(data_home).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / filename
    if not cache_path.exists():
        urlretrieve(url, cache_path)
    return cache_path


def _synthetic_entry(
    name: str,
    sampler: DatasetSampler,
    *,
    description: str,
    tail_index_alpha: Any = None,
    dim: int | None = None,
    sampler_kwargs_builders: SamplerKwargBuilders | None = None,
) -> DatasetEntry:
    """Build one synthetic dataset registry entry."""
    return DatasetEntry(
        name=name,
        dataset_type="synthetic",
        description=description,
        tail_index_alpha=tail_index_alpha,
        split_mode="random",
        dim=dim,
        sampler=sampler,
        sampler_kwargs_builders=sampler_kwargs_builders or {},
    )


def _alpha_stable_mixture_base_scale(kwargs: dict[str, Any]) -> float:
    """Read the alpha-stable mixture scale, accepting base_std as a legacy analogue."""
    if "base_scale" in kwargs:
        return float(getpop(kwargs, "base_scale"))
    return float(getpop(kwargs, "base_std", 0.55))


def _real_entry(
    name: str,
    loader: DatasetLoader,
    *,
    description: str,
    tail_index_alpha: Any = None,
    split_mode: str = "random",
    standardize_default: bool = True,
    dim: int | tuple[int, ...] | None = None,
    n_samples: int | None = None,
) -> DatasetEntry:
    """Build one real dataset registry entry."""
    return DatasetEntry(
        name=name,
        dataset_type="real",
        description=description,
        tail_index_alpha=tail_index_alpha,
        split_mode=split_mode,
        standardize_default=standardize_default,
        dim=dim,
        n_samples=n_samples,
        loader=loader,
    )


def _resolve_dataset(
    target_data: str,
    all_datasets: dict[str, DatasetEntry],
    *,
    dataset_type: str | None = None,
) -> DatasetEntry:
    """Resolve one dataset entry and optionally enforce its type."""
    key = target_data.lower()
    if key not in all_datasets:
        raise KeyError(f"Unknown dataset {target_data!r}. Available datasets: {sorted(all_datasets)}")

    entry = all_datasets[key]
    if dataset_type is not None and entry.dataset_type != dataset_type:
        raise ValueError(
            f"Dataset {target_data!r} is {entry.dataset_type!r}, expected {dataset_type!r}."
        )
    return entry


def _standardize_split_arrays(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize validation and test splits using train statistics."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (
        (x_train - mean) / std,
        (x_val - mean) / std,
        (x_test - mean) / std,
    )


def to_tensor_triplet(
    x_train: np.ndarray | torch.Tensor,
    x_val: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert train, val, and test arrays to tensors."""
    return (
        torch.as_tensor(_copy_if_numpy(x_train), device=device, dtype=dtype),
        torch.as_tensor(_copy_if_numpy(x_val), device=device, dtype=dtype),
        torch.as_tensor(_copy_if_numpy(x_test), device=device, dtype=dtype),
    )


def coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw frame to a numeric matrix."""
    numeric_frame = frame.copy()
    numeric_frame = numeric_frame.dropna(axis=0).reset_index(drop=True)

    for column in numeric_frame.columns:
        if pd.api.types.is_datetime64_any_dtype(numeric_frame[column]):
            numeric_frame[column] = numeric_frame[column].astype("int64") // 10**9

    non_numeric_columns = [
        column for column in numeric_frame.columns if not pd.api.types.is_numeric_dtype(numeric_frame[column])
    ]
    if non_numeric_columns:
        numeric_frame = pd.get_dummies(
            numeric_frame,
            columns=non_numeric_columns,
            drop_first=False,
        )

    numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
    numeric_frame = numeric_frame.dropna(axis=0).reset_index(drop=True)
    if numeric_frame.shape[1] == 0:
        raise ValueError("No numeric columns remain after preprocessing.")
    return numeric_frame.astype(float)


def split_frame(
    frame: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a frame into train, val, and test arrays."""
    train_idx, val_idx, test_idx = split_sample_indices(
        len(frame),
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    return (
        frame.iloc[train_idx].reset_index(drop=True).to_numpy(),
        frame.iloc[val_idx].reset_index(drop=True).to_numpy(),
        frame.iloc[test_idx].reset_index(drop=True).to_numpy(),
    )


def split_sample_indices(
    n_rows: int,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices into train, val, and test partitions."""
    if not 0 <= val_size < 1:
        raise ValueError("val_size must lie in [0, 1).")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must lie in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    if n_rows < 3:
        raise ValueError("The dataset must contain at least 3 rows.")
    if split_mode not in {"random", "chronological"}:
        raise ValueError("split_mode must be 'random' or 'chronological'.")

    if split_mode == "random":
        indices = np.arange(n_rows)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        return train_idx, val_idx, test_idx

    n_test = int(np.floor(test_size * n_rows))
    n_val = int(np.floor(val_size * n_rows))
    n_train = n_rows - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("The requested val/test proportions leave an empty split.")
    return (
        np.arange(0, n_train),
        np.arange(n_train, n_train + n_val),
        np.arange(n_train + n_val, n_rows),
    )


def _split_tensorize_frame(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one frame and return train, val, and test tensors."""
    x_train, x_val, x_test = split_frame(
        frame,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )

    if standardize:
        x_train, x_val, x_test = _standardize_split_arrays(x_train, x_val, x_test)

    return to_tensor_triplet(x_train, x_val, x_test, device=device, dtype=dtype)


def split_tensor_data(
    data: torch.Tensor | np.ndarray,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one tensor-backed dataset along its leading sample dimension."""
    x = torch.as_tensor(_copy_if_numpy(data))
    if x.ndim < 1:
        raise ValueError(f"Expected at least one sample dimension, got shape {tuple(x.shape)}.")
    n_rows = int(x.shape[0])
    if not 0 <= val_size < 1:
        raise ValueError("val_size must lie in [0, 1).")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must lie in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    if n_rows < 3:
        raise ValueError("The dataset must contain at least 3 samples.")
    if split_mode not in {"random", "chronological"}:
        raise ValueError("split_mode must be 'random' or 'chronological'.")

    if split_mode == "random":
        indices = np.arange(n_rows)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        train_idx = torch.as_tensor(train_idx, dtype=torch.long)
        val_idx = torch.as_tensor(val_idx, dtype=torch.long)
        test_idx = torch.as_tensor(test_idx, dtype=torch.long)
        return x.index_select(0, train_idx), x.index_select(0, val_idx), x.index_select(0, test_idx)

    n_test = int(np.floor(test_size * n_rows))
    n_val = int(np.floor(val_size * n_rows))
    n_train = n_rows - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("The requested val/test proportions leave an empty split.")
    return x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]


def _load_wildfires(**kwargs: Any) -> pd.DataFrame:
    """Load the wildfire size dataset."""
    data_home = kwargs.pop("data_home", _resolve_real_data_home() / "powerlaws")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected wildfire loader kwargs: {unexpected}.")
    source = _cache_remote_text_file(
        "https://aaronclauset.github.io/powerlaws/data/fires.txt",
        data_home=data_home,
        filename="fires.txt",
    )
    frame = pd.read_csv(
        source,
        header=None,
        sep=r"\s+",
    )
    if frame.shape[1] == 1:
        return pd.DataFrame({"acres_burned": frame.iloc[:, 0].astype(float)})
    frame.columns = [f"x{i}" for i in range(frame.shape[1] - 1)] + ["acres_burned"]
    return frame.astype(float)


def _load_earthquakes(**kwargs: Any) -> pd.DataFrame:
    """Load the earthquake magnitude dataset."""
    data_home = kwargs.pop("data_home", _resolve_real_data_home() / "powerlaws")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected earthquake loader kwargs: {unexpected}.")
    source = _cache_remote_text_file(
        "https://aaronclauset.github.io/powerlaws/data/quakes.txt",
        data_home=data_home,
        filename="quakes.txt",
    )
    frame = pd.read_csv(
        source,
        header=None,
        sep=r"\s+",
    )
    return pd.DataFrame({"magnitude": frame.iloc[:, 0].astype(float)})


def _load_kddcup(**kwargs: Any) -> pd.DataFrame:
    """Load the KDD Cup 99 intrusion dataset as numeric tabular features."""
    data_home = kwargs.pop("data_home", _resolve_real_data_home() / "scikit_learn")
    bunch = fetch_kddcup99(
        as_frame=True,
        percent10=True,
        data_home=str(data_home),
        **kwargs,
    )

    features = bunch.data.copy()
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    for column in features.columns:
        if pd.api.types.is_object_dtype(features[column]):
            features[column] = features[column].map(_decode_byte_string)

    target = bunch.target.copy()
    if isinstance(target, pd.Series) and pd.api.types.is_object_dtype(target):
        target = target.map(_decode_byte_string)

    numeric_features = features.select_dtypes(include=[np.number]).copy()
    if numeric_features.shape[1] == 0:
        raise ValueError("KDD Cup 99 loader produced no numeric feature columns.")
    return numeric_features


def _load_default_credit(**kwargs: Any) -> pd.DataFrame:
    """Load the Default of Credit Card Clients dataset as tabular features."""
    data_home = kwargs.pop("data_home", _resolve_real_data_home() / "scikit_learn")
    bunch = fetch_openml(
        data_id=42477,
        as_frame=True,
        parser="pandas",
        data_home=str(data_home),
        **kwargs,
    )

    frame = bunch.frame.copy()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    target = None
    target_names = bunch.target_names
    if isinstance(target_names, str) and target_names in frame.columns:
        target = frame.pop(target_names)
    elif isinstance(target_names, list):
        matching_target_names = [name for name in target_names if name in frame.columns]
        if matching_target_names:
            target = frame.pop(matching_target_names[0])

    if target is None:
        target = bunch.target.copy()

    if "ID" in frame.columns:
        frame = frame.drop(columns=["ID"])

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        if coerced.notna().sum() == frame[column].notna().sum():
            frame[column] = coerced

    if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
        target = target.iloc[:, 0]
    if isinstance(target, pd.Series) and not pd.api.types.is_numeric_dtype(target):
        coerced_target = pd.to_numeric(target, errors="coerce")
        if coerced_target.notna().sum() == target.notna().sum():
            target = coerced_target

    return frame


_LVIS_ANNOTATION_FILES: dict[str, tuple[str, ...]] = {
    "train": ("lvis_v1_train.json", "lvis_v0.5_train.json"),
    "val": ("lvis_v1_val.json", "lvis_v0.5_val.json"),
}
_LVIS_IMAGE_DIR_NAMES = ("train2017", "val2017")


def _normalize_lvis_split(split: str) -> str:
    """Normalize LVIS split aliases to annotation filename suffixes."""
    split_key = str(split).lower().strip()
    if split_key in {"validation", "valid"}:
        split_key = "val"
    if split_key not in _LVIS_ANNOTATION_FILES:
        raise ValueError(f"Unsupported LVIS split {split!r}. Expected one of {sorted(_LVIS_ANNOTATION_FILES)}.")
    return split_key


def _lvis_base_roots(data_home: str | Path | None) -> list[Path]:
    """Resolve user-provided LVIS root or Jean Zay DSDIR roots."""
    if data_home is not None:
        roots = [Path(data_home).expanduser()]
    else:
        dsdir = os.getenv("DSDIR")
        if not dsdir:
            raise RuntimeError(
                "LVIS data_home was not provided and $DSDIR is not set. "
                "This loader is Jean-Zay/local-files only and will not download LVIS assets."
            )
        roots = [Path(dsdir).expanduser()]

    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        raise RuntimeError(f"LVIS search root(s) do not exist: {missing}")
    return roots


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Return existing paths in first-seen order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            key = expanded.resolve()
        except OSError:
            key = expanded.absolute()
        if key in seen or not expanded.exists():
            continue
        seen.add(key)
        unique.append(expanded)
    return unique


def _lvis_search_roots(base_roots: list[Path], annotation_path: Path | None = None) -> list[Path]:
    """Build common LVIS/COCO search roots without assuming one fixed Jean Zay layout."""
    roots: list[Path] = []
    if annotation_path is not None:
        roots.extend([annotation_path.parent, annotation_path.parent.parent])
    for root in base_roots:
        roots.extend(
            [
                root / "annotations",
                root / "lvis",
                root / "LVIS",
                root / "lvis" / "annotations",
                root / "LVIS" / "annotations",
                root / "coco",
                root / "COCO",
                root / "coco" / "annotations",
                root / "COCO" / "annotations",
                root / "images",
                root / "coco" / "images",
                root / "COCO" / "images",
                root,
            ]
        )
    return _unique_paths(roots)


def _find_named_paths(
    roots: list[Path],
    names: set[str],
    *,
    want_dir: bool,
    max_depth: int = 6,
) -> list[Path]:
    """Find files or directories with selected names under roots, bounded by depth."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            if not want_dir and root.name in names:
                resolved = root.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(root)
            continue

        if want_dir and root.name in names:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(root)

        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                depth = 0
            if depth >= max_depth:
                dirs[:] = []
            dirs[:] = [name for name in dirs if not name.startswith(".") and name != "__pycache__"]

            candidates = dirs if want_dir else files
            for name in candidates:
                if name not in names:
                    continue
                path = current_path / name
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append(path)
    return found


def _find_lvis_annotation(base_roots: list[Path], split: str) -> tuple[Path, list[Path]]:
    """Locate the LVIS annotation file for one split."""
    names = set(_LVIS_ANNOTATION_FILES[split])
    roots = _lvis_search_roots(base_roots)
    matches = _find_named_paths(roots, names, want_dir=False)
    if matches:
        return matches[0], roots

    searched_roots = "\n".join(f"  - {root}" for root in roots)
    searched_files = ", ".join(sorted(names))
    raise RuntimeError(
        "Could not find LVIS annotation file. "
        f"Searched for {searched_files} under:\n{searched_roots}"
    )


def _load_lvis_json(annotation_path: Path) -> dict[str, Any]:
    """Load one local LVIS annotation JSON file."""
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"LVIS annotation file {annotation_path} did not decode to a JSON object.")
    if not isinstance(data.get("images"), list):
        raise RuntimeError(f"LVIS annotation file {annotation_path} has no 'images' list.")
    return data


def _lvis_candidate_images(
    data: dict[str, Any],
    *,
    category_frequency: str | None,
    annotation_path: Path,
) -> list[dict[str, Any]]:
    """Return deduplicated LVIS image records after optional category-frequency filtering."""
    images = data.get("images", [])
    images_by_id: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []
    for image in images:
        if not isinstance(image, dict) or "id" not in image:
            continue
        image_id = int(image["id"])
        if image_id in images_by_id:
            continue
        images_by_id[image_id] = image
        ordered_ids.append(image_id)

    if category_frequency is None:
        return [images_by_id[image_id] for image_id in ordered_ids]

    frequency = str(category_frequency).lower().strip()
    if frequency not in {"r", "c", "f"}:
        raise ValueError("category_frequency must be one of 'r', 'c', 'f', or None.")

    categories = data.get("categories", [])
    category_frequency_by_id: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict) or "id" not in category:
            continue
        label = category.get("frequency", category.get("freq"))
        if label is not None:
            category_frequency_by_id[int(category["id"])] = str(label).lower()

    if not category_frequency_by_id:
        raise RuntimeError(
            f"category_frequency={frequency!r} was requested, but {annotation_path} "
            "does not expose LVIS category frequency labels."
        )

    matching_ids: set[int] = set()
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        category_id = annotation.get("category_id")
        image_id = annotation.get("image_id")
        if category_id is None or image_id is None:
            continue
        if category_frequency_by_id.get(int(category_id)) == frequency:
            matching_ids.add(int(image_id))

    return [images_by_id[image_id] for image_id in ordered_ids if image_id in matching_ids]


def _lvis_category_maps(data: dict[str, Any]) -> tuple[dict[int, str], dict[int, str]]:
    """Return LVIS category-name and category-frequency maps."""
    name_by_id: dict[int, str] = {}
    frequency_by_id: dict[int, str] = {}
    for category in data.get("categories", []):
        if not isinstance(category, dict) or "id" not in category:
            continue
        category_id = int(category["id"])
        name_by_id[category_id] = str(category.get("name", category_id))
        label = category.get("frequency", category.get("freq"))
        if label is not None:
            frequency_by_id[category_id] = str(label).lower()
    return name_by_id, frequency_by_id


def _lvis_category_ids_by_image(data: dict[str, Any]) -> dict[int, list[int]]:
    """Return unique LVIS category ids per image, preserving first annotation order."""
    categories_by_image: dict[int, list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id is None or category_id is None:
            continue
        image_id = int(image_id)
        category_id = int(category_id)
        pair = (image_id, category_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        categories_by_image.setdefault(image_id, []).append(category_id)
    return categories_by_image


def _lvis_record(
    image: dict[str, Any],
    image_path: Path,
    category_ids: list[int],
    *,
    name_by_id: dict[int, str],
    frequency_by_id: dict[int, str],
) -> dict[str, Any]:
    """Build one metadata record aligned with an LVIS image tensor."""
    file_name = str(image.get("file_name") or image_path.name)
    return {
        "image_id": int(image["id"]),
        "file_name": file_name,
        "path": str(image_path),
        "category_ids": [int(category_id) for category_id in category_ids],
        "category_names": [name_by_id.get(int(category_id), str(category_id)) for category_id in category_ids],
        "category_frequencies": [
            frequency_by_id[int(category_id)]
            for category_id in category_ids
            if int(category_id) in frequency_by_id
        ],
    }


def _select_lvis_images(
    images: list[dict[str, Any]],
    *,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Subsample image records before loading pixels."""
    if max_samples is None or len(images) <= max_samples:
        return images
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(images), size=max_samples, replace=False))
    return [images[int(idx)] for idx in selected]


def _find_lvis_image_dirs(search_roots: list[Path]) -> dict[str, list[Path]]:
    """Find common COCO image folders used by LVIS annotations."""
    dirs = _find_named_paths(search_roots, set(_LVIS_IMAGE_DIR_NAMES), want_dir=True)
    by_name: dict[str, list[Path]] = {name: [] for name in _LVIS_IMAGE_DIR_NAMES}
    for path in dirs:
        by_name.setdefault(path.name, []).append(path)
    return by_name


def _lvis_url_parts(value: Any) -> tuple[str | None, str | None]:
    """Extract image folder and filename from a COCO URL-like LVIS field."""
    if not value:
        return None, None
    parts = str(value).rstrip("/").split("/")
    if len(parts) < 2:
        return None, Path(parts[-1]).name if parts else None
    return parts[-2], Path(parts[-1]).name


def _resolve_lvis_image_path(
    image: dict[str, Any],
    *,
    split: str,
    image_dirs: dict[str, list[Path]],
    search_roots: list[Path],
) -> Path:
    """Resolve one LVIS image record to an existing local image path."""
    file_name_raw = image.get("file_name")
    file_name = str(file_name_raw) if file_name_raw else ""
    file_path = Path(file_name) if file_name else None
    basename = file_path.name if file_path is not None and file_path.name else ""
    url_folder, url_name = _lvis_url_parts(image.get("coco_url"))
    if not basename:
        basename = url_name or f"{int(image['id']):012d}.jpg"

    candidates: list[Path] = []
    if file_path is not None:
        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            candidates.extend(root / file_path for root in search_roots)

    folder_names = [url_folder, f"{split}2017", *_LVIS_IMAGE_DIR_NAMES]
    seen_folders: set[str] = set()
    for folder_name in folder_names:
        if not folder_name or folder_name in seen_folders:
            continue
        seen_folders.add(folder_name)
        for image_dir in image_dirs.get(folder_name, []):
            candidates.append(image_dir / basename)

    for dirs in image_dirs.values():
        candidates.extend(image_dir / basename for image_dir in dirs)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {root}" for root in search_roots)
    raise RuntimeError(
        f"Could not resolve LVIS image for image id={image.get('id')} file_name={file_name!r}. "
        f"Searched image roots:\n{searched}"
    )


def _read_lvis_image(path: Path, image_size: int) -> torch.Tensor:
    """Read one local image as float32 RGB CHW tensor in [0, 1]."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("LVIS loader requires Pillow (PIL) to read local image files.") from exc

    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_lvis(**kwargs: Any) -> DatasetPayload:
    """Load pre-fetched Jean Zay LVIS images as a local tensor dataset."""
    split = _normalize_lvis_split(str(kwargs.pop("split", "train")))
    image_size = int(kwargs.pop("image_size", 64))
    max_samples_raw = kwargs.pop("max_samples", None)
    max_samples = None if max_samples_raw is None else int(max_samples_raw)
    category_frequency_raw = kwargs.pop("category_frequency", None)
    category_frequency = None if category_frequency_raw is None else str(category_frequency_raw)
    seed = int(kwargs.pop("seed", 0))
    data_home = kwargs.pop("data_home", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected LVIS loader kwargs: {unexpected}.")
    if image_size <= 0:
        raise ValueError("image_size must be > 0.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    base_roots = _lvis_base_roots(data_home)
    annotation_path, _ = _find_lvis_annotation(base_roots, split)
    data = _load_lvis_json(annotation_path)
    images = _lvis_candidate_images(
        data,
        category_frequency=category_frequency,
        annotation_path=annotation_path,
    )
    images = _select_lvis_images(images, max_samples=max_samples, seed=seed)
    if not images:
        raise RuntimeError(
            f"No LVIS images matched split={split!r} category_frequency={category_frequency!r} "
            f"in {annotation_path}."
        )

    search_roots = _lvis_search_roots(base_roots, annotation_path=annotation_path)
    image_dirs = _find_lvis_image_dirs(search_roots)
    name_by_id, frequency_by_id = _lvis_category_maps(data)
    category_ids_by_image = _lvis_category_ids_by_image(data)
    tensors = []
    records = []
    for image in images:
        image_path = _resolve_lvis_image_path(
            image,
            split=split,
            image_dirs=image_dirs,
            search_roots=search_roots,
        )
        tensors.append(_read_lvis_image(image_path, image_size))
        image_id = int(image["id"])
        records.append(
            _lvis_record(
                image,
                image_path,
                category_ids_by_image.get(image_id, []),
                name_by_id=name_by_id,
                frequency_by_id=frequency_by_id,
            )
        )
    return DatasetPayload(
        data=torch.stack(tensors, dim=0).to(dtype=torch.float32),
        metadata={
            "source_split": split,
            "annotation_path": str(annotation_path),
            "image_size": int(image_size),
            "category_frequency_filter": category_frequency,
            "n_selected_images": len(records),
            "records": records,
        },
    )


_hrrr_thread_local = threading.local()


def _hrrr_session() -> requests.Session:
    if not hasattr(_hrrr_thread_local, "session"):
        _hrrr_thread_local.session = requests.Session()
    return _hrrr_thread_local.session


def _hrrr_apcp_byte_range(url: str, session: requests.Session, forecast_hour: int) -> tuple[int, int | None] | None:
    """Return (start, end) byte range for the APCP accumulation field, or None if unavailable."""
    try:
        resp = session.get(url + ".idx", timeout=60)
        resp.raise_for_status()
    except Exception:
        return None
    lines = resp.text.strip().splitlines()
    needle = f":APCP:surface:0-{int(forecast_hour)} hour acc"
    for i, line in enumerate(lines):
        if needle in line:
            start = int(line.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            return start, end
    return None


def _parse_hrrr_datetime(value: str) -> datetime:
    """Parse one HRRR timestamp and normalize it to UTC."""
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hrrr_cache_path(
    cache_root: Path,
    *,
    start: datetime,
    end: datetime,
    bbox: tuple[float, float, float, float],
    step_hours: int,
    forecast_hours: tuple[int, ...],
) -> Path:
    """Build a cache filename that changes when the requested HRRR grid changes."""
    payload = {
        "version": 2,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox": [round(float(value), 5) for value in bbox],
        "step_hours": int(step_hours),
        "forecast_hours": [int(value) for value in forecast_hours],
        "field": "APCP",
        "shape": [1, 100, 100],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return cache_root / (
        f"hrrr_apcp_100x100_{start:%Y%m%d%H}_{end:%Y%m%d%H}"
        f"_step{int(step_hours):02d}_f{'-'.join(f'{hour:02d}' for hour in forecast_hours)}_{digest}.pt"
    )


def _hrrr_fetch_one(
    dt: datetime,
    tmp_root: Path,
    lon_w: float,
    lon_e: float,
    lat_s: float,
    lat_n: float,
    forecast_hours: tuple[int, ...],
) -> np.ndarray | None:
    """Download, subset, and decode one HRRR APCP timestamp. Returns None on any failure."""
    session = _hrrr_session()
    ymd = dt.strftime("%Y%m%d")
    hh = dt.strftime("%H")
    for forecast_hour in forecast_hours:
        url = (
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{int(forecast_hour):02d}.grib2"
        )
        tag = f"{ymd}{hh}_f{int(forecast_hour):02d}"
        raw_path = tmp_root / f"_raw_{tag}.grib2"
        cut_path = tmp_root / f"_cut_{tag}.grib2"
        try:
            byte_range = _hrrr_apcp_byte_range(url, session, forecast_hour)
            if byte_range is None:
                continue
            start_byte, end_byte = byte_range
            range_header = f"bytes={start_byte}-{end_byte}" if end_byte is not None else f"bytes={start_byte}-"
            resp = session.get(url, headers={"Range": range_header}, timeout=120, stream=True)
            resp.raise_for_status()
            with open(raw_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)
            subprocess.run(
                ["wgrib2", str(raw_path), "-small_grib", f"{lon_w}:{lon_e}", f"{lat_s}:{lat_n}", str(cut_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with xr.open_dataset(cut_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                var_name = next(iter(ds.data_vars))
                values = np.asarray(ds.data_vars[var_name].squeeze(drop=True).values, dtype=np.float32)
            height, width = values.shape
            if height < 100 or width < 100:
                continue
            i0 = (height - 100) // 2
            j0 = (width - 100) // 2
            return np.nan_to_num(values[i0:i0 + 100, j0:j0 + 100], nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            continue
        finally:
            raw_path.unlink(missing_ok=True)
            cut_path.unlink(missing_ok=True)
    return None


def _load_hrrr(**kwargs: Any) -> torch.Tensor:
    """Load HRRR accumulated precipitation fields as one image-like tensor."""
    data_home_arg = kwargs.pop("data_home", None)
    out_dir_arg = kwargs.pop("out_dir", None)
    start = str(kwargs.pop("start", "2017-01-01 00:00:00"))
    end = str(kwargs.pop("end", "2025-09-30 18:00:00"))
    bbox = tuple(kwargs.pop("bbox", (-96.0, -91.5, 28.5, 32.0)))
    step_hours = int(kwargs.pop("step_hours", 1))
    forecast_hours_raw = kwargs.pop("forecast_hours", None)
    forecast_hour_raw = kwargs.pop("forecast_hour", 1)
    n_workers = int(kwargs.pop("n_workers", 32))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected HRRR loader kwargs: {unexpected}.")
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain four values, got {bbox}.")
    bbox = tuple(float(value) for value in bbox)
    if step_hours <= 0:
        raise ValueError(f"step_hours must be positive, got {step_hours}.")
    if forecast_hours_raw is None:
        forecast_hours = (int(forecast_hour_raw),)
    elif isinstance(forecast_hours_raw, (list, tuple)):
        forecast_hours = tuple(int(value) for value in forecast_hours_raw)
    else:
        forecast_hours = (int(forecast_hours_raw),)
    if not forecast_hours:
        raise ValueError("forecast_hours must contain at least one forecast hour.")
    if any(hour <= 0 for hour in forecast_hours):
        raise ValueError(f"forecast_hours must be positive, got {forecast_hours}.")

    cache_root_arg = data_home_arg if data_home_arg is not None else out_dir_arg
    cache_root = Path(cache_root_arg).expanduser() if cache_root_arg is not None else _resolve_real_data_home() / "hrrr"
    cache_root.mkdir(parents=True, exist_ok=True)

    t0 = _parse_hrrr_datetime(start)
    t1 = _parse_hrrr_datetime(end)
    if t1 < t0:
        raise ValueError(f"end must be >= start, got {end!r} < {start!r}.")
    total_seconds = int((t1 - t0).total_seconds())
    step_seconds = int(step_hours * 3600)
    if total_seconds % step_seconds != 0:
        raise ValueError(f"start/end must be spaced on a {step_hours}-hour HRRR grid.")

    cache_path = _hrrr_cache_path(
        cache_root,
        start=t0,
        end=t1,
        bbox=bbox,
        step_hours=step_hours,
        forecast_hours=forecast_hours,
    )
    cache_candidates = [cache_path]
    if (start == "2018-01-01 00:00:00" and end == "2025-09-30 18:00:00" and step_hours == 6 and forecast_hours == (6,)):
        cache_candidates.extend([cache_root / "hrrr.pt", cache_root / "hrrr_apcp_100x100.pt"])
    for path in cache_candidates:
        if not path.exists():
            continue
        cached = torch.as_tensor(torch.load(path, map_location="cpu"))
        if cached.ndim == 3:
            cached = cached.unsqueeze(1)
        if cached.ndim != 4 or tuple(cached.shape[1:]) != (1, 100, 100):
            raise ValueError(f"Unexpected cached HRRR tensor shape {tuple(cached.shape)} in {path}.")
        cached = cached.to(device="cpu", dtype=torch.float32)
        if path != cache_path:
            torch.save(cached, cache_path)
        return cached

    if shutil.which("wgrib2") is None:
        raise RuntimeError("HRRR loader requires wgrib2.")

    timestamps: list[datetime] = []
    current = t0
    while current <= t1:
        timestamps.append(current)
        current += timedelta(hours=step_hours)

    lon_w, lon_e, lat_s, lat_n = bbox
    frame_map: dict[datetime, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="flowbench-hrrr-", dir=str(cache_root)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_hrrr_fetch_one, dt, tmp_root, lon_w, lon_e, lat_s, lat_n, forecast_hours): dt
                for dt in timestamps
            }
            with tqdm(total=len(timestamps), desc="Fetching HRRR") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        frame_map[futures[future]] = result
                    pbar.update(1)

    if not frame_map:
        raise RuntimeError(
            "No HRRR frames could be fetched. "
            f"Requested {len(timestamps)} timestamps from {start!r} to {end!r} "
            f"with step_hours={step_hours} and forecast_hours={forecast_hours}."
        )
    if len(frame_map) < len(timestamps):
        warnings.warn(
            f"Fetched {len(frame_map)} / {len(timestamps)} HRRR frames; missing timestamps were skipped.",
            stacklevel=2,
        )

    frames = [frame_map[dt] for dt in sorted(frame_map)]
    fields = np.stack(frames, axis=0)[:, np.newaxis]
    tensor = torch.from_numpy(fields)
    torch.save(tensor, cache_path)
    return tensor


ALL_DATASETS: dict[str, DatasetEntry] = {
    "unbalanced_highdim_gaussian_mixture": _synthetic_entry(
        "unbalanced_highdim_gaussian_mixture",
        sample_unbalanced_highdim_gaussian_mixture,
        description="Imbalanced high-dimensional Gaussian mixture synthetic dataset.",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(getpop(kwargs, "dim", 50)),
            "n_modes": lambda kwargs: int(getpop(kwargs, "n_modes", 16)),
            "rank": lambda kwargs: int(getpop(kwargs, "rank", 6)),
            "imbalance_tau": lambda kwargs: float(getpop(kwargs, "imbalance_tau", 1.2)),
            "mean_scale": lambda kwargs: float(getpop(kwargs, "mean_scale", 7.5)),
            "base_std": lambda kwargs: float(getpop(kwargs, "base_std", 0.55)),
            "anisotropy": lambda kwargs: float(getpop(kwargs, "anisotropy", 1.0)),
            "structure_seed": lambda kwargs: int(getpop(kwargs, "structure_seed", 0)),
        },
    ),
    "unbalanced_highdim_alpha_stable_mixture": _synthetic_entry(
        "unbalanced_highdim_alpha_stable_mixture",
        sample_unbalanced_highdim_alpha_stable_mixture,
        description="Imbalanced high-dimensional mixture with alpha-stable local noise.",
        tail_index_alpha="configurable",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(getpop(kwargs, "dim", 50)),
            "alpha": lambda kwargs: float(getpop(kwargs, "alpha", 1.7)),
            "n_modes": lambda kwargs: int(getpop(kwargs, "n_modes", 16)),
            "rank": lambda kwargs: int(getpop(kwargs, "rank", 6)),
            "imbalance_tau": lambda kwargs: float(getpop(kwargs, "imbalance_tau", 1.2)),
            "mean_scale": lambda kwargs: float(getpop(kwargs, "mean_scale", 7.5)),
            "base_scale": _alpha_stable_mixture_base_scale,
            "anisotropy": lambda kwargs: float(getpop(kwargs, "anisotropy", 1.0)),
            "structure_seed": lambda kwargs: int(getpop(kwargs, "structure_seed", 0)),
        },
    ),
    "gaussian": _synthetic_entry(
        "gaussian",
        sample_gaussian,
        description="Isotropic Gaussian synthetic dataset.",
        tail_index_alpha=2.0,
        sampler_kwargs_builders={"dim": lambda kwargs: int(getpop(kwargs, "dim", 1))},
    ),
    "checker": _synthetic_entry(
        "checker",
        sample_checker,
        description="Checkerboard synthetic dataset.",
        dim=2,
    ),
    "spiral": _synthetic_entry(
        "spiral",
        sample_spiral,
        description="Noisy spiral synthetic dataset.",
        dim=2,
        sampler_kwargs_builders={
            "spiral_turns": lambda kwargs: float(getpop(kwargs, "spiral_turns", 3.0)),
            "spiral_radius": lambda kwargs: float(getpop(kwargs, "spiral_radius", 4.0)),
            "spiral_noise": lambda kwargs: float(getpop(kwargs, "spiral_noise", 0.2)),
        },
    ),
    "alpha_stable": _synthetic_entry(
        "alpha_stable",
        sample_scaled_isotropic_alpha_stable,
        description="Isotropic alpha-stable synthetic dataset.",
        tail_index_alpha="configurable",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(getpop(kwargs, "dim", 1)),
            "alpha": lambda kwargs: float(getpop(kwargs, "alpha", 1.99)),
        },
    ),
    "student": _synthetic_entry(
        "student",
        sample_student_t,
        description="Student-t synthetic dataset.",
        tail_index_alpha="configurable",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(getpop(kwargs, "dim", 1)),
            "nu": lambda kwargs: float(getpop(kwargs, "nu", 10.0)),
        },
    ),
    "exponential": _synthetic_entry(
        "exponential",
        sample_exponential,
        description="Exponential synthetic dataset.",
        sampler_kwargs_builders={
            "dim": lambda kwargs: int(getpop(kwargs, "dim", 1)),
            "rate": lambda kwargs: float(getpop(kwargs, "rate", 1.0)),
        },
    ),
    "wildfires": _real_entry(
        "wildfires",
        _load_wildfires,
        description="U.S. wildfire sizes in acres.",
        tail_index_alpha=(1.1, 1.8),
        dim=1,
    ),
    "earthquakes": _real_entry(
        "earthquakes",
        _load_earthquakes,
        description="Earthquake magnitude benchmark from the Clauset collection.",
        dim=1,
    ),
    "kddcup": _real_entry(
        "kddcup",
        _load_kddcup,
        description="KDD Cup 99 intrusion dataset with numeric feature columns only.",
    ),
    "default_credit": _real_entry(
        "default_credit",
        _load_default_credit,
        description="Default of Credit Card Clients dataset from OpenML.",
    ),
    "lvis": _real_entry(
        "lvis",
        _load_lvis,
        description="LVIS long-tailed object categories from local Jean Zay COCO/LVIS files; default image_size=64.",
        standardize_default=False,
        dim=(3, 64, 64),
    ),
    "hrrr": _real_entry(
        "hrrr",
        _load_hrrr,
        description="HRRR accumulated precipitation fields on a 100x100 crop.",
        standardize_default=False,
        dim=(1, 100, 100),
    ),
}

SYNTHETIC_DATASETS: dict[str, DatasetEntry] = {
    name: entry for name, entry in ALL_DATASETS.items() if entry.dataset_type == "synthetic"
}

REAL_DATASETS: dict[str, DatasetEntry] = {
    name: entry for name, entry in ALL_DATASETS.items() if entry.dataset_type == "real"
}
