"""Benchmark 1 figure/table creation from saved results."""

import argparse
import json
from pathlib import Path
import yaml
from cauda.table import save_double_entry_table
from results_utils import resolve_run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    parser.add_argument("--table-root", type=Path, default=Path("_tables"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.out_root, "bench_1", args.run_dir)
    table_dir = args.table_root / "bench_1" / run_dir.name
    table_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((run_dir / "config.yml").read_text(encoding="utf-8"))
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    results = {}
    for item in payload["metrics"]:
        results[(item["model"], item["metric"])] = item["values"]

    col_order = ["GaussianFlowLinear", "AlphaStableFlowLinear"]
    col_name = {
        "GaussianFlowLinear": "Lipman et al",
        "AlphaStableFlowLinear": "Our approach",
    }
    metric_name = {"MSLE": r"$10 \\times \\mathrm{MSLE}_{\\xi=0.95}$"}
    caption = (
        r"Comparison of generative models on $\\alpha$-stable synthetic data ($\\alpha = "
        f"{float(cfg['alpha']):.2f}$)."
    )

    filename = save_double_entry_table(
        results=results,
        plot_dir=str(table_dir),
        fmt="{:.2f}",
        caption=caption,
        label=f"tab:synthetic_alpha_{float(cfg['alpha']):.2f}",
        col_order=col_order,
        metric_direction={"MSLE": "down"},
        col_name=col_name,
        metric_name=metric_name,
        show_metric_arrows=True,
        bold_best_in_row=True,
        suffix=f"_{cfg['target_data_type']}_comparison",
    )
    print(f"[INFO] Saved table artifact: {filename}")
