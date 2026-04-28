"""LVIS-specific dataset helpers."""

from typing import Any


def prepare_lvis_loader_kwargs(
    entry_name: str,
    kwargs: dict[str, Any],
    n_samples: int | None,
) -> tuple[dict[str, Any], int | None]:
    """Map n_samples to LVIS max_samples when the caller did not set it already."""
    if entry_name != "lvis" or n_samples is None or "max_samples" in kwargs:
        return kwargs, n_samples

    loader_kwargs = dict(kwargs)
    loader_kwargs["max_samples"] = n_samples
    return loader_kwargs, None
