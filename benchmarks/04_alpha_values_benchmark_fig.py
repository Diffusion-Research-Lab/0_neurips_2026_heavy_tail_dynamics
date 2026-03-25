"""Benchmark 2 figure creation from saved results."""

import argparse
import json
from pathlib import Path

from genkit.plotting import plot_heatmap

from _utils import resolve_run_dir


BENCHMARK_NAME = "bench_2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    parser.add_argument("--fig-root", type=Path, default=Path("_figures"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.out_root, BENCHMARK_NAME, args.run_dir)
    fig_dir = args.fig_root / BENCHMARK_NAME / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    grid = payload.get("grid", [])
    if not grid:
        raise ValueError("results.json does not contain any grid entries.")

    metric_names = ["MSSLE_95"]
    available_metrics = set(grid[0].get("metrics", {}).keys())
    missing_metrics = [metric_name for metric_name in metric_names if metric_name not in available_metrics]
    if missing_metrics:
        raise ValueError(f"results.json is missing required metrics: {missing_metrics}")

    for metric_name in metric_names:
        results = {
            (float(item["alpha_data"]), float(item["alpha_model"])): item["metrics"][metric_name]["values"]
            for item in grid
        }

        filename = plot_heatmap(
            results=results,
            plot_dir=str(fig_dir),
            xlabel=r"$\alpha$-model",
            ylabel=r"$\alpha$-data",
            fontsize=16,
            suffix=metric_name.lower(),
        )
        print(f"[INFO] Saved figure artifact: {filename}")
