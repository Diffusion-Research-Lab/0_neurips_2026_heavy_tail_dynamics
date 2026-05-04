"""Public dataset API."""

from ._dataset import (
    fetch_real_data,
    fetch_synthetic_data,
    get_dataset_metadata,
    list_datasets,
)

__all__ = [
    "fetch_real_data",
    "fetch_synthetic_data",
    "get_dataset_metadata",
    "list_datasets",
]
