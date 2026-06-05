"""Public real-dataset API."""

__all__ = [
    "fetch_real_data",
    "get_dataset_metadata",
    "list_datasets",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import _dataset

    return getattr(_dataset, name)
