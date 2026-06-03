"""Public real-dataset API."""

from ._dataset import (
    fetch_real_data,
    get_dataset_metadata,
    list_datasets,
)

__all__ = [
    "fetch_real_data",
    "get_dataset_metadata",
    "list_datasets",
]
