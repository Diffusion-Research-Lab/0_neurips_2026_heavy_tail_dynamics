"""Shared image I/O helpers for image dataset loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


_IMAGE_LOADERS_USING_MAX_SAMPLES = frozenset({"lvis", "cifar100_lt", "imagenet_lt"})


def prepare_image_loader_kwargs(
    entry_name: str,
    kwargs: dict[str, Any],
    n_samples: int | None,
) -> tuple[dict[str, Any], int | None]:
    """Map ``n_samples`` to ``max_samples`` for image loaders that subsample by class."""
    if entry_name not in _IMAGE_LOADERS_USING_MAX_SAMPLES:
        return kwargs, n_samples
    if n_samples is None or "max_samples" in kwargs:
        return kwargs, n_samples
    loader_kwargs = dict(kwargs)
    loader_kwargs["max_samples"] = n_samples
    return loader_kwargs, None


def _resampling() -> Any:
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def array_uint8_to_resized_tensor(array: np.ndarray, image_size: int) -> torch.Tensor:
    """Convert an ``(H, W, 3)`` uint8 array to a ``(3, image_size, image_size)`` float tensor."""
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 uint8 image array, got shape {array.shape}.")
    image = Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), _resampling())
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def read_rgb_resized(path: Path, image_size: int) -> torch.Tensor:
    """Read an image as RGB, resize to ``image_size``, return a CHW float tensor in ``[0, 1]``."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), _resampling())
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
