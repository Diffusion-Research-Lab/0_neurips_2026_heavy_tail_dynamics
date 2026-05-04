"""Benchmark helpers and entrypoints.

This package marker is kept intentionally lazy: the numbered entrypoints
``01_main.py`` and ``02_evaluate.py`` import torch and other heavy deps, so
they are *not* imported at package-load time. Consumers should call
``importlib.import_module("benchmarks.01_main")`` (etc.) directly when they
need them. Lightweight modules (``benchmarks.utils``,
``benchmarks._real_data_cache`` for cache-key helpers, etc.) can be imported
without paying the torch import cost.
"""
