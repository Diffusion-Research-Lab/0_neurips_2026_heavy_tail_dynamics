"""Private dataset helpers and registries."""

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_kddcup99, fetch_openml
from sklearn.model_selection import train_test_split
from .utils import getpop
from ._sampling import (
    sample_balanced_bimodal_gaussian,
    sample_checker,
    sample_exponential,
    sample_gaussian,
    sample_scaled_isotropic_alpha_stable,
    sample_spiral,
    sample_student_t,
    sample_unbalanced_highdim_gaussian_mixture,
    sample_unbalanced_bimodal_gaussian,
)


DatasetLoader = Callable[..., pd.DataFrame]
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


def _synthetic_entry(
    name: str,
    sampler: DatasetSampler,
    *,
    description: str,
    tail_index_alpha: Any = None,
    sampler_kwargs_builders: SamplerKwargBuilders | None = None,
) -> DatasetEntry:
    """Build one synthetic dataset registry entry."""
    return DatasetEntry(
        name=name,
        dataset_type="synthetic",
        description=description,
        tail_index_alpha=tail_index_alpha,
        split_mode="random",
        sampler=sampler,
        sampler_kwargs_builders=sampler_kwargs_builders or {},
    )


def _real_entry(
    name: str,
    loader: DatasetLoader,
    *,
    description: str,
    tail_index_alpha: Any = None,
    split_mode: str = "random",
) -> DatasetEntry:
    """Build one real dataset registry entry."""
    return DatasetEntry(
        name=name,
        dataset_type="real",
        description=description,
        tail_index_alpha=tail_index_alpha,
        split_mode=split_mode,
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
    if not 0 <= val_size < 1:
        raise ValueError("val_size must lie in [0, 1).")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must lie in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    if len(frame) < 3:
        raise ValueError("The dataset must contain at least 3 rows.")
    if split_mode not in {"random", "chronological"}:
        raise ValueError("split_mode must be 'random' or 'chronological'.")

    if split_mode == "random":
        train_frame, test_frame = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        val_ratio = val_size / (1.0 - test_size)
        train_frame, val_frame = train_test_split(
            train_frame,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        train_frame = train_frame.reset_index(drop=True)
        val_frame = val_frame.reset_index(drop=True)
        test_frame = test_frame.reset_index(drop=True)
    else:
        n_rows = len(frame)
        n_test = int(np.floor(test_size * n_rows))
        n_val = int(np.floor(val_size * n_rows))
        n_train = n_rows - n_val - n_test
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError("The requested val/test proportions leave an empty split.")
        train_frame = frame.iloc[:n_train].reset_index(drop=True)
        val_frame = frame.iloc[n_train: n_train + n_val].reset_index(drop=True)
        test_frame = frame.iloc[n_train + n_val:].reset_index(drop=True)

    return train_frame.to_numpy(), val_frame.to_numpy(), test_frame.to_numpy()


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


def _load_wildfires(**_: Any) -> pd.DataFrame:
    """Load the wildfire size dataset."""
    frame = pd.read_csv(
        "https://aaronclauset.github.io/powerlaws/data/fires.txt",
        header=None,
        sep=r"\s+",
    )
    if frame.shape[1] == 1:
        return pd.DataFrame({"acres_burned": frame.iloc[:, 0].astype(float)})
    frame.columns = [f"x{i}" for i in range(frame.shape[1] - 1)] + ["acres_burned"]
    return frame.astype(float)


def _load_earthquakes(**_: Any) -> pd.DataFrame:
    """Load the earthquake magnitude dataset."""
    frame = pd.read_csv(
        "https://aaronclauset.github.io/powerlaws/data/quakes.txt",
        header=None,
        sep=r"\s+",
    )
    return pd.DataFrame({"magnitude": frame.iloc[:, 0].astype(float)})


def _load_kddcup99(**kwargs: Any) -> pd.DataFrame:
    """Load the KDD Cup 99 intrusion dataset as numeric tabular features."""
    data_home = kwargs.pop("data_home", "/tmp/scikit_learn_data")
    bunch = fetch_kddcup99(
        as_frame=True,
        percent10=True,
        data_home=data_home,
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
    data_home = kwargs.pop("data_home", "/tmp/scikit_learn_data")
    bunch = fetch_openml(
        data_id=42477,
        as_frame=True,
        parser="pandas",
        data_home=data_home,
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


ALL_DATASETS: dict[str, DatasetEntry] = {
    "balanced_bimodal_gaussian": _synthetic_entry(
        "balanced_bimodal_gaussian",
        sample_balanced_bimodal_gaussian,
        description="Balanced bimodal Gaussian synthetic dataset.",
        sampler_kwargs_builders={"dim": lambda kwargs: int(getpop(kwargs, "dim", 1))},
    ),
    "unbalanced_bimodal_gaussian": _synthetic_entry(
        "unbalanced_bimodal_gaussian",
        sample_unbalanced_bimodal_gaussian,
        description="Legacy alias for an imbalanced high-dimensional Gaussian mixture synthetic dataset.",
        sampler_kwargs_builders={"dim": lambda kwargs: int(getpop(kwargs, "dim", 2))},
    ),
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
    ),
    "spiral": _synthetic_entry(
        "spiral",
        sample_spiral,
        description="Noisy spiral synthetic dataset.",
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
    ),
    "earthquakes": _real_entry(
        "earthquakes",
        _load_earthquakes,
        description="Earthquake magnitude benchmark from the Clauset collection.",
    ),
    "kddcup99": _real_entry(
        "kddcup99",
        _load_kddcup99,
        description="KDD Cup 99 intrusion dataset with numeric feature columns only.",
    ),
    "default_credit": _real_entry(
        "default_credit",
        _load_default_credit,
        description="Default of Credit Card Clients dataset from OpenML.",
    ),
}

SYNTHETIC_DATASETS: dict[str, DatasetEntry] = {
    name: entry for name, entry in ALL_DATASETS.items() if entry.dataset_type == "synthetic"
}

REAL_DATASETS: dict[str, DatasetEntry] = {
    name: entry for name, entry in ALL_DATASETS.items() if entry.dataset_type == "real"
}
