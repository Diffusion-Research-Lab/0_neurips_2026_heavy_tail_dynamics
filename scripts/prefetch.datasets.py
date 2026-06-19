"""Build processed real-dataset split caches for benchmark jobs."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    sys.path.insert(0, str(path))

from benchmarks.utils import load_yaml, require_section, select_entries  # noqa: E402

# Must mirror benchmarks/_real_data_cache.CACHE_VERSION. Tested for equality.
CACHE_VERSION = 1

DEFAULT_CONFIGS = [
    "benchmarks/configs/pilot/image.yaml",
    "benchmarks/configs/templates/image_bench.yaml",
    "benchmarks/configs/bench/image",
    "benchmarks/configs/viz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="FlowBench dataset-cache root. Defaults to $FLOWBENCH_DATA, then $WORK/flowbench_data.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing processed caches.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate caches without building. Exit 0 if all variants are cached, 2 otherwise. "
             "Torch-free; safe to run on a login node without loading the JZ pytorch module.",
    )
    parser.add_argument(
        "--only-dataset",
        action="append",
        default=[],
        help="Restrict preprocessing to one real dataset name. May be passed multiple times.",
    )
    parser.add_argument("configs", nargs="*", type=Path, help="Benchmark configs to scan.")
    return parser.parse_args()


def _expand_config_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if not path.exists() and not path.suffix:
            continue
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.yaml")))
        else:
            expanded.append(path)
    return expanded


def _size_label(dataset_cfg: dict) -> str:
    params = dataset_cfg.get("params", {})
    if "max_samples" in params:
        return f"max_samples={params['max_samples']}"
    if "n_samples" in params:
        return f"n_samples={params['n_samples']}"
    return "full"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dtype_str(dtype_value) -> str:
    if isinstance(dtype_value, str):
        return dtype_value.lower().replace("torch.", "").strip()
    return str(dtype_value).replace("torch.", "")


def cache_key_from_yaml(dataset_cfg: dict, dtype_value) -> str:
    """Reproduce ``benchmarks._real_data_cache.real_dataset_cache_key`` torch-free."""
    request = {
        "cache_version": CACHE_VERSION,
        "kind": "real",
        "name": str(dataset_cfg.get("name", "")).strip(),
        "params": _jsonable(copy.deepcopy(dataset_cfg.get("params", {}))),
        "split": _jsonable(copy.deepcopy(dataset_cfg.get("split", {}))),
        "dtype": _dtype_str(dtype_value),
    }
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cache_path_from_yaml(root: Path, dataset_cfg: dict, dtype_value) -> Path:
    name = str(dataset_cfg["name"]).strip()
    key = cache_key_from_yaml(dataset_cfg, dtype_value)
    return Path(root) / "processed" / name / f"{name}_{key}.pt"


def _resolve_cache_root(cache_root_arg) -> Path:
    if cache_root_arg is not None:
        return Path(cache_root_arg).expanduser()
    flowbench_data = os.environ.get("FLOWBENCH_DATA")
    if flowbench_data:
        return Path(flowbench_data).expanduser()
    work = os.environ.get("WORK")
    if work:
        return Path(work).expanduser() / "flowbench_data"
    home = os.environ.get("HOME")
    if home:
        return Path(home).expanduser() / ".cache" / "flowbench_data"
    return Path("/tmp") / "flowbench_data"


def _real_dataset_variants_yaml(config_path: Path) -> list[tuple[dict, str]]:
    """Return ``(dataset_cfg, dtype_str)`` per real-dataset variant in ``config_path``."""
    config = load_yaml(config_path)
    dtype_str = str(config["run"].get("dtype", "float64"))
    sweep = require_section(config, "sweep")
    datasets = select_entries("datasets", require_section(config, "datasets"), list(sweep["datasets"]))
    variants = []
    for dataset in datasets:
        dataset_cfg = dataset["config"]
        if dataset_cfg.get("kind") != "real":
            continue
        variants.append(({**dataset_cfg, "preset_name": dataset["variant_name"]}, dtype_str))
    return variants


def _collect_yaml_tasks(args) -> list[tuple[Path, dict, str]]:
    config_inputs = args.configs or [PROJECT_ROOT / path for path in DEFAULT_CONFIGS]
    configs = _expand_config_inputs(config_inputs)
    only_datasets = {name.strip() for name in args.only_dataset if name.strip()}
    seen_keys: set[str] = set()
    tasks: list[tuple[Path, dict, str]] = []
    for config_path in configs:
        for dataset_cfg, dtype_str in _real_dataset_variants_yaml(config_path):
            if only_datasets and dataset_cfg["name"] not in only_datasets:
                continue
            key = cache_key_from_yaml(dataset_cfg, dtype_str)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            tasks.append((config_path, dataset_cfg, dtype_str))
    return tasks


def _check_only(tasks: list, root: Path) -> int:
    print(f"[dataset] check-only: {len(tasks)} variant(s) under {root / 'processed'}", flush=True)
    missing: list[str] = []
    for index, (config_path, dataset_cfg, dtype_str) in enumerate(tasks, start=1):
        name = str(dataset_cfg["name"])
        label = _size_label(dataset_cfg)
        cache_path = cache_path_from_yaml(root, dataset_cfg, dtype_str)
        if not cache_path.is_file():
            print(
                f"[MISSING] {index}/{len(tasks)} {name:14s} {label:16s} "
                f"preset={dataset_cfg['preset_name']} config={config_path.name} path={cache_path}",
                flush=True,
            )
            missing.append(name)
            continue
        size_mb = cache_path.stat().st_size / (1024 ** 2)
        print(
            f"[VALID]   {index}/{len(tasks)} {name:14s} {label:16s} "
            f"size={size_mb:.1f}MB path={cache_path}",
            flush=True,
        )
    if missing:
        print(
            f"[CHECK] {len(missing)}/{len(tasks)} variant(s) need preprocessing "
            f"({', '.join(sorted(set(missing)))})",
            flush=True,
        )
        return 2
    print(f"[CHECK] all {len(tasks)} variant(s) already preprocessed; no sbatch needed", flush=True)
    return 0


def _run_build(args, tasks: list, root: Path) -> int:
    # Heavy imports happen here; require torch/sklearn/labkit available.
    import time
    from benchmarks._real_data_cache import (
        CACHE_VERSION as RUNTIME_CACHE_VERSION,
        build_preprocessed_real_dataset,
    )
    from labkit.config import parse_dtype

    if RUNTIME_CACHE_VERSION != CACHE_VERSION:
        raise RuntimeError(
            f"CACHE_VERSION drift: prefetch.datasets={CACHE_VERSION} "
            f"vs benchmarks._real_data_cache={RUNTIME_CACHE_VERSION}"
        )

    print(f"[dataset] processed dataset-cache root: {root / 'processed'}")
    print(f"[dataset] variants: {len(tasks)}")
    for index, (config_path, dataset_cfg, dtype_str) in enumerate(tasks, start=1):
        dtype = parse_dtype(dtype_str)
        started_at = time.perf_counter()
        print(
            f"[dataset] start  {index}/{len(tasks)} "
            f"{dataset_cfg['name']:12s} {_size_label(dataset_cfg):16s} "
            f"preset={dataset_cfg['preset_name']} config={config_path.name}",
            flush=True,
        )
        result = build_preprocessed_real_dataset(
            dataset_cfg,
            dtype,
            data_root=root,
            overwrite=args.overwrite,
            source_config=config_path,
        )
        elapsed = time.perf_counter() - started_at
        print(
            f"[dataset] {result['status']:6s} {index}/{len(tasks)} "
            f"{dataset_cfg['name']:12s} {_size_label(dataset_cfg):16s} "
            f"elapsed={elapsed:.1f}s path={result['path']}",
            flush=True,
        )
    print(f"[dataset] ready: {len(tasks)} real dataset variant(s)")
    return 0


if __name__ == "__main__":

    args = parse_args()
    root = _resolve_cache_root(args.cache_root)
    tasks = _collect_yaml_tasks(args)
    if args.check_only:
        raise SystemExit(_check_only(tasks, root))
    raise SystemExit(_run_build(args, tasks, root))
