#!/usr/bin/env python
"""Prefetch all real FlowBench datasets into a persistent cache root."""

import argparse
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genkit._datasets import _resolve_real_data_home
from genkit.datasets import fetch_real_data, list_datasets, get_dataset_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache all FlowBench real datasets.")
    parser.add_argument("--data-root", type=Path, default=None, help="Cache root. Defaults to $WORK/flowbench/data, then $HOME/.cache/flowbench/data.")
    return parser.parse_args()


def _real_dataset_names() -> list[str]:
    return [
        name
        for name in list_datasets()
        if get_dataset_metadata(name)["dataset_type"] == "real"
    ]


def _dataset_kwargs(name: str, *, data_root: Path) -> dict[str, str]:
    if name == "hrrr":
        return {"data_home": str(data_root / "hrrr")}
    if name in {"default_credit", "kddcup"}:
        return {"data_home": str(data_root / "scikit_learn")}
    return {"data_home": str(data_root / "powerlaws")}


def _shape_summary(data: object) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
    if isinstance(data, tuple):
        return tuple(tuple(split.shape) for split in data)
    return tuple(data.shape)


def main() -> int:
    args = _parse_args()
    data_root = args.data_root.expanduser() if args.data_root is not None else _resolve_real_data_home()
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[prefetch] caching real datasets under {data_root}")
    for name in _real_dataset_names():
        kwargs = _dataset_kwargs(name, data_root=data_root)
        print(f"[prefetch] {name} ...", flush=True)
        data = fetch_real_data(name, **kwargs)
        print(f"[prefetch] {name} ready shape={_shape_summary(data)} cache={next(iter(kwargs.values()))}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
