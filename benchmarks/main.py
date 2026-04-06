#!/usr/bin/env python
"""Minimal configurable benchmark entrypoint."""

import argparse
import itertools
from pathlib import Path
import pandas as pd
from genkit.datasets import list_datasets
from benchmarks._utils import load_yaml, make_batch_dir, require_section, resolve_dtype, run_one, select_entries, setup_logging


if __name__ == "__main__":

    setup_logging()

    parser = argparse.ArgumentParser(description="Minimal configurable benchmark entrypoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to one YAML config file.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_cfg = require_section(config, "run")
    sweep_cfg = require_section(config, "sweep")
    datasets_cfg = require_section(config, "datasets")
    networks_cfg = require_section(config, "networks")
    models_cfg = require_section(config, "models")
    trains_cfg = require_section(config, "trains")
    save_cfg = require_section(config, "save")

    dtype = resolve_dtype(str(run_cfg.get("dtype", "float32")))
    dataset_variants = select_entries("datasets", datasets_cfg, list(sweep_cfg.get("datasets", [])))
    network_variants = select_entries("networks", networks_cfg, list(sweep_cfg.get("networks", [])))
    model_variants = select_entries("models", models_cfg, list(sweep_cfg.get("models", [])))
    train_variants = select_entries("trains", trains_cfg, list(sweep_cfg.get("trains", [])))

    if not dataset_variants or not network_variants or not model_variants or not train_variants:
        raise ValueError("The sweep must select at least one dataset, network, model, and train entry.")

    batch_dir = make_batch_dir(run_cfg=run_cfg, save_cfg=save_cfg)
    print(f"batch_dir: {batch_dir}")
    print(f"available datasets: {', '.join(list_datasets())}")

    manifest_rows = []
    combinations = itertools.product(dataset_variants, network_variants, model_variants, train_variants)
    for combo_index, (dataset_variant, network_variant, model_variant, train_variant) in enumerate(combinations, start=1):
        combo_name = "__".join(
            [
                dataset_variant["variant_name"],
                network_variant["variant_name"],
                model_variant["variant_name"],
                train_variant["variant_name"],
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
                save_cfg=save_cfg,
                dtype=dtype,
            )
        )

    pd.DataFrame(manifest_rows).to_csv(batch_dir / "manifest.csv", index=False)
    with (batch_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"n_runs: {len(manifest_rows)}\n")
        handle.write(f"config: {args.config.resolve()}\n")
        handle.write(f"batch_dir: {batch_dir.resolve()}\n")
    print("done")
