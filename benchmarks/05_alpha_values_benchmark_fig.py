"""Benchmark 2 figure creation from saved results."""

import argparse
import json
from pathlib import Path
from cauda.plotting import plot_heatmap
from results_utils import resolve_run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    parser.add_argument("--fig-root", type=Path, default=Path("_figures"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.out_root, "bench_2", args.run_dir)
    fig_dir = args.fig_root / "bench_2" / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    results = {}
    for item in payload["grid"]:
        results[(float(item["alpha_data"]), float(item["alpha_model"]))] = item["values"]

    filename = plot_heatmap(
        results=results,
        plot_dir=str(fig_dir),
        xlabel=r"$\alpha$-model",
        ylabel=r"$\alpha$-data",
        fontsize=16,
    )
    print(f"[INFO] Saved figure artifact: {filename}")
