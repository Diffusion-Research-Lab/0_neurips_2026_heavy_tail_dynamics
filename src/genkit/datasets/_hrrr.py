"""HRRR-specific dataset helpers."""

from typing import Any


def prepare_hrrr_loader_kwargs(
    entry_name: str,
    kwargs: dict[str, Any],
    n_samples: int | None,
) -> tuple[dict[str, Any], int | None]:
    """Pass n_samples through to the HRRR loader so it can resume on disk."""
    if entry_name != "hrrr" or n_samples is None:
        return kwargs, n_samples

    loader_kwargs = dict(kwargs)
    loader_kwargs["n_samples"] = n_samples
    return loader_kwargs, None
