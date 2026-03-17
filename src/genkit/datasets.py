"""Sampling functions for various distributions."""

import warnings
import torch
from .utils import getpop
from ._sampling import (sample_balanced_bimodal_gaussian, sample_unbalanced_bimodal_gaussian,
                        sample_gaussian, sample_scaled_isotropic_alpha_stable, sample_student_t,
                        sample_spiral, sample_exponential)


def fetch_synthetic_data(target_data: str, **kwargs):
    """
    Sample (X_train, X_test) from a chosen synthetic target distribution.
    """
    n_samples = int(getpop(kwargs, "n_samples", 10_000))
    device = getpop(kwargs, "device", 'cpu')
    dtype = getpop(kwargs, "dtype", torch.float32)
    kwargs_sampling = dict(n_samples=n_samples, device=device, dtype=dtype)

    if target_data == "balanced_bimodal_gaussian":
        kwargs_sampling['dim'] = int(getpop(kwargs, "dim", 1))
        X_train = sample_balanced_bimodal_gaussian(**kwargs_sampling)
        X_val = sample_balanced_bimodal_gaussian(**kwargs_sampling)
        X_test = sample_balanced_bimodal_gaussian(**kwargs_sampling)

    elif target_data == "unbalanced_bimodal_gaussian":
        kwargs_sampling['dim'] = int(getpop(kwargs, "dim", 1))
        X_train = sample_unbalanced_bimodal_gaussian(**kwargs_sampling)
        X_val = sample_unbalanced_bimodal_gaussian(**kwargs_sampling)
        X_test = sample_unbalanced_bimodal_gaussian(**kwargs_sampling)

    elif target_data == "gaussian":
        kwargs_sampling['dim'] = int(getpop(kwargs, "dim", 1))
        X_train = sample_gaussian(**kwargs_sampling)
        X_val = sample_gaussian(**kwargs_sampling)
        X_test = sample_gaussian(**kwargs_sampling)

    elif target_data == "spiral":
        kwargs_sampling['spiral_turns'] = float(getpop(kwargs, "spiral_turns", 3.0))
        kwargs_sampling['spiral_radius'] = float(getpop(kwargs, "spiral_radius", 4.0))
        kwargs_sampling['spiral_noise'] = float(getpop(kwargs, "spiral_noise", 0.2))
        X_train = sample_spiral(**kwargs_sampling)
        X_val = sample_spiral(**kwargs_sampling)
        X_test = sample_spiral(**kwargs_sampling)

    elif target_data == "alpha_stable":
        kwargs_sampling['dim'] = int(getpop(kwargs, "dim", 1))
        kwargs_sampling['alpha'] = float(getpop(kwargs, "alpha", 1.99))
        X_train = sample_scaled_isotropic_alpha_stable(**kwargs_sampling)
        X_val = sample_scaled_isotropic_alpha_stable(**kwargs_sampling)
        X_test = sample_scaled_isotropic_alpha_stable(**kwargs_sampling)

    elif target_data == "student":
        kwargs_sampling['dim'] = int(getpop(kwargs, "dim", 1))
        kwargs_sampling['nu'] = float(getpop(kwargs, "nu", 10.0))
        X_train = sample_student_t(**kwargs_sampling)
        X_val = sample_student_t(**kwargs_sampling)
        X_test = sample_student_t(**kwargs_sampling)

    elif target_data == "exponential":
        kwargs_sampling['dim'] = int(getpop(kwargs, "dim", 1))
        kwargs_sampling['rate'] = float(getpop(kwargs, "rate", 1.0))
        X_train = sample_exponential(**kwargs_sampling)
        X_val = sample_exponential(**kwargs_sampling)
        X_test = sample_exponential(**kwargs_sampling)

    else:
        raise ValueError(f"target_data not understood, got {target_data!r}")

    if len(kwargs) > 0:
        warnings.warn(f"In 'fetch_synthetic_data', {kwargs.keys()} arguments were ignored.")

    return X_train, X_val, X_test
