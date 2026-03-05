"""Benchmark 3 figure creation from saved results."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from results_utils import resolve_run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    parser.add_argument("--fig-root", type=Path, default=Path("_figures"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.out_root, "bench_3", args.run_dir)
    fig_dir = args.fig_root / "bench_3" / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    models = list(payload["results"].keys())
    cov = [payload["results"][m]["minority_coverage_at_k"] for m in models]
    mass = [payload["results"][m]["minority_mass_ratio"] for m in models]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(models, cov)
    ax[0].set_title("Minority Coverage@k")
    ax[0].set_ylim(0.0, 1.05)
    ax[0].tick_params(axis="x", rotation=20)

    ax[1].bar(models, mass)
    ax[1].set_title("Minority Mass Ratio")
    ax[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out = fig_dir / "bench_3_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved figure artifact: {out}")
