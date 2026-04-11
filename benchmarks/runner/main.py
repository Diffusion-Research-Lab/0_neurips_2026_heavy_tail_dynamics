#!/usr/bin/env python
"""Minimal configurable benchmark entrypoint."""

import argparse
import itertools
from pathlib import Path
import pandas as pd
from genkit.datasets import list_datasets
from labkit.config import parse_dtype
try:
    from benchmarks.runner._utils import load_yaml, make_batch_dir, require_section, run_one, select_entries, setup_logging
except ModuleNotFoundError:
    from _utils import load_yaml, make_batch_dir, require_section, run_one, select_entries, setup_logging


if __name__ == "__main__":

    setup_logging()

    parser = argparse.ArgumentParser(description="Minimal configurable benchmark entrypoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to one YAML config file.")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Optional existing/shared batch directory.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split the combination grid into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to execute.")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")

    config = load_yaml(args.config)
    run_cfg = require_section(config, "run")
    sweep_cfg = require_section(config, "sweep")
    datasets_cfg = require_section(config, "datasets")
    networks_cfg = require_section(config, "networks")
    models_cfg = require_section(config, "models")
    trains_cfg = require_section(config, "trains")
    save_cfg = require_section(config, "save")

    dtype = parse_dtype(str(run_cfg.get("dtype", "float32")))
    dataset_variants = select_entries("datasets", datasets_cfg, list(sweep_cfg.get("datasets", [])))
    network_variants = select_entries("networks", networks_cfg, list(sweep_cfg.get("networks", [])))
    model_variants = select_entries("models", models_cfg, list(sweep_cfg.get("models", [])))
    train_variants = select_entries("trains", trains_cfg, list(sweep_cfg.get("trains", [])))
    n_trial = int(run_cfg.get("n_trial", 1))

    if not dataset_variants or not network_variants or not model_variants or not train_variants:
        raise ValueError("The sweep must select at least one dataset, network, model, and train entry.")
    if n_trial < 1:
        raise ValueError("run.n_trial must be >= 1.")

    if args.batch_dir is None:
        batch_dir = make_batch_dir(run_cfg=run_cfg, save_cfg=save_cfg)
    else:
        batch_dir = args.batch_dir.expanduser()
        batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"batch_dir: {batch_dir}")
    print(f"shard: {args.shard_index + 1}/{args.shard_count}")
    print(f"available datasets: {', '.join(list_datasets())}")

    manifest_rows = []
    combinations = list(itertools.product(dataset_variants, network_variants, model_variants, train_variants, range(n_trial)))
    for combo_index, (dataset_variant, network_variant, model_variant, train_variant, trial_idx) in enumerate(combinations, start=1):
        if (combo_index - 1) % args.shard_count != args.shard_index:
            continue
        combo_name = "__".join(
            [
                dataset_variant["variant_name"],
                network_variant["variant_name"],
                model_variant["variant_name"],
                train_variant["variant_name"],
                f"trial-{trial_idx + 1:02d}",
            ]
        )
        manifest_rows.append(
            run_one(
                config_path=args.config,
                batch_dir=batch_dir,
                combo_index=combo_index,
                combo_name=combo_name,
                run_cfg=run_cfg,
                dataset_variant=dataset_variant,
                network_variant=network_variant,
                model_variant=model_variant,
                train_variant=train_variant,
                trial_idx=trial_idx,
                save_cfg=save_cfg,
                dtype=dtype,
            )
        )

    manifest_name = "manifest.csv" if args.shard_count == 1 else f"manifest_shard_{args.shard_index:03d}.csv"
    summary_name = "summary.txt" if args.shard_count == 1 else f"summary_shard_{args.shard_index:03d}.txt"
    pd.DataFrame(manifest_rows).to_csv(batch_dir / manifest_name, index=False)
    n_failed = sum(row.get("status") == "failed" for row in manifest_rows)
    with (batch_dir / summary_name).open("w", encoding="utf-8") as handle:
        handle.write(f"n_runs: {len(manifest_rows)}\n")
        handle.write(f"n_failed: {n_failed}\n")
        handle.write(f"config: {args.config.resolve()}\n")
        handle.write(f"batch_dir: {batch_dir.resolve()}\n")
        handle.write(f"shard_index: {args.shard_index}\n")
        handle.write(f"shard_count: {args.shard_count}\n")
    print("done")
