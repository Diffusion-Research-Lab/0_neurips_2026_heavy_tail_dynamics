#!/usr/bin/env python
"""Report training/evaluation settings from benchmark artifacts."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any
import yaml


DEFAULT_ARTIFACT_ROOT = Path("benchmarks/artifacts")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def stable_key(value: Any) -> str:
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))


def add_unique(target: list[Any], value: Any) -> None:
    if value in (None, "", []):
        return
    key = stable_key(value)
    if all(stable_key(existing) != key for existing in target):
        target.append(normalize(value))


def pick(mapping: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: normalize(mapping[key]) for key in keys if key in mapping}


def first_dim(shape: Any) -> int | None:
    if isinstance(shape, (list, tuple)) and shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    return None


def source_config_path(eval_run_dir: Path, eval_summary: dict[str, Any]) -> Path | None:
    source_run_dir = eval_summary.get("source_run_dir")
    if source_run_dir:
        path = Path(str(source_run_dir)) / "config.yaml"
        if path.is_file():
            return path

    batch_dir = eval_run_dir.parent
    if batch_dir.name.endswith("_evaluate"):
        path = batch_dir.with_name(batch_dir.name.removesuffix("_evaluate")) / eval_run_dir.name / "config.yaml"
        if path.is_file():
            return path
    return None


def finite_metric_counts(scalars_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        with gzip.open(scalars_path, mode="rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                source = row.get("source", "")
                metric = row.get("metric_name", "")
                if not metric:
                    continue
                key = f"{source}:{metric}" if source else metric
                counts[key]["rows"] += 1
                try:
                    value = float(row.get("value", ""))
                except ValueError:
                    value = float("nan")
                if math.isfinite(value):
                    counts[key]["finite"] += 1
    except OSError:
        return {}
    return {metric: dict(counter) for metric, counter in sorted(counts.items())}


def summarize_dataset(dataset_name: str, eval_run_dirs: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "n_eval_runs": len(eval_run_dirs),
        "train_samples": [],
        "val_samples": [],
        "test_split_samples": [],
        "eval_samples": [],
        "feature_dim": [],
        "data": {
            "kind": [],
            "name": [],
            "params": [],
            "split": [],
            "resolved_shapes": [],
        },
        "training": {
            "dtype": [],
            "network": [],
            "models": [],
            "train_params": [],
            "n_trials": [],
        },
        "evaluation": {
            "n_eval_repeats": [],
            "sample_batch_size": [],
            "requested_n_eval_samples": [],
            "eval_reference_source": [],
            "max_mmd_samples": [],
            "max_tv_samples": [],
            "checkpoint_epochs": [],
            "metric_names": [],
        },
        "metric_value_counts": {},
        "warnings": Counter(),
    }

    metric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for eval_run_dir in eval_run_dirs:
        eval_summary = load_yaml(eval_run_dir / "summary.yaml")
        config_path = source_config_path(eval_run_dir, eval_summary)
        config = load_yaml(config_path) if config_path else {}

        dataset_cfg = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
        run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
        network_cfg = config.get("network") if isinstance(config.get("network"), dict) else {}
        model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
        train_cfg = config.get("train") if isinstance(config.get("train"), dict) else {}

        train_shape = dataset_cfg.get("resolved_train_shape")
        val_shape = dataset_cfg.get("resolved_val_shape")
        test_shape = dataset_cfg.get("resolved_test_shape")
        add_unique(result["train_samples"], first_dim(train_shape))
        add_unique(result["val_samples"], first_dim(val_shape))
        add_unique(result["test_split_samples"], eval_summary.get("test_split_samples") or first_dim(test_shape))
        add_unique(result["eval_samples"], eval_summary.get("n_eval_samples"))
        add_unique(result["feature_dim"], eval_summary.get("feature_dim"))

        add_unique(result["data"]["kind"], dataset_cfg.get("kind"))
        add_unique(result["data"]["name"], dataset_cfg.get("name"))
        add_unique(result["data"]["params"], dataset_cfg.get("params", {}))
        add_unique(result["data"]["split"], dataset_cfg.get("split", {}))
        add_unique(
            result["data"]["resolved_shapes"],
            {
                "train": train_shape,
                "val": val_shape,
                "test": test_shape,
            },
        )

        add_unique(result["training"]["dtype"], run_cfg.get("resolved_dtype") or run_cfg.get("dtype"))
        add_unique(
            result["training"]["network"],
            {
                "preset": network_cfg.get("preset_name"),
                "name": network_cfg.get("name"),
                "params": network_cfg.get("params", {}),
                "param_count": network_cfg.get("resolved_param_count"),
            },
        )
        add_unique(
            result["training"]["models"],
            {
                "preset": model_cfg.get("preset_name"),
                "name": model_cfg.get("name"),
                "params": model_cfg.get("params", {}),
            },
        )
        add_unique(
            result["training"]["train_params"],
            pick(
                train_cfg,
                [
                    "preset_name",
                    "device",
                    "data_device",
                    "batch_size",
                    "n_epochs",
                    "lr",
                    "use_adamw",
                    "weight_decay",
                    "grad_clip_norm",
                    "lr_schedule",
                    "warmup_steps",
                    "cosine_eta_min_ratio",
                    "freq_logging",
                    "ckpt_freq_epochs",
                    "ckpt_keep_last",
                    "num_workers",
                    "pin_memory",
                ],
            ),
        )
        add_unique(result["training"]["n_trials"], run_cfg.get("n_trial"))

        for key in [
            "n_eval_repeats",
            "sample_batch_size",
            "requested_n_eval_samples",
            "eval_reference_source",
            "max_mmd_samples",
            "max_tv_samples",
            "checkpoint_epochs",
            "metric_names",
        ]:
            add_unique(result["evaluation"][key], eval_summary.get(key))

        for warning in eval_summary.get("warnings") or []:
            result["warnings"][str(warning)] += 1

        for metric, counts in finite_metric_counts(eval_run_dir / "scalars.csv.gz").items():
            metric_counts[metric].update(counts)

    result["warnings"] = dict(result["warnings"].most_common())
    result["metric_value_counts"] = {metric: dict(counts) for metric, counts in sorted(metric_counts.items())}
    return result


def discover_eval_runs(artifact_root: Path) -> dict[str, list[Path]]:
    datasets: dict[str, list[Path]] = defaultdict(list)
    for eval_batch_dir in sorted(path for path in artifact_root.glob("*_evaluate") if path.is_dir()):
        if "pilot" in eval_batch_dir.name.lower():
            continue
        for eval_run_dir in sorted(path for path in eval_batch_dir.iterdir() if path.is_dir() and path.name[:3].isdigit()):
            summary = load_yaml(eval_run_dir / "summary.yaml")
            config_path = source_config_path(eval_run_dir, summary)
            config = load_yaml(config_path) if config_path else {}
            dataset_cfg = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
            dataset = dataset_cfg.get("preset_name") or dataset_cfg.get("name")
            if not dataset:
                scalars_path = eval_run_dir / "scalars.csv.gz"
                try:
                    with gzip.open(scalars_path, mode="rt", encoding="utf-8", newline="") as handle:
                        row = next(csv.DictReader(handle), None)
                    if row:
                        dataset = row.get("dataset_preset") or row.get("dataset_name")
                except OSError:
                    dataset = None
            if dataset:
                datasets[str(dataset)].append(eval_run_dir)
    return dict(sorted(datasets.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print benchmark data/training/evaluation settings as JSON.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset presets to include.")
    parser.add_argument("--no-metric-counts", action="store_true", help="Drop metric finite-value counts from the output.")
    parser.add_argument("--no-warnings", action="store_true", help="Drop warning counts from the output.")
    args = parser.parse_args()

    eval_runs_by_dataset = discover_eval_runs(args.artifact_root)
    selected = set(args.datasets or eval_runs_by_dataset)
    report = {
        "artifact_root": str(args.artifact_root),
        "datasets": [
            summarize_dataset(dataset, eval_runs)
            for dataset, eval_runs in eval_runs_by_dataset.items()
            if dataset in selected
        ],
    }
    for dataset_report in report["datasets"]:
        if args.no_metric_counts:
            dataset_report.pop("metric_value_counts", None)
        if args.no_warnings:
            dataset_report.pop("warnings", None)
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
