"""Benchmark 3 table creation from saved results."""

import argparse
import json
from pathlib import Path
import yaml
from results_utils import resolve_run_dir


def _fmt(x: float) -> str:
    return f"{float(x):.4f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    parser.add_argument("--table-root", type=Path, default=Path("_tables"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.out_root, "bench_3", args.run_dir)
    table_dir = args.table_root / "bench_3" / run_dir.name
    table_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((run_dir / "config.yml").read_text(encoding="utf-8"))
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    model_order = ["GaussianFlowLinear", "AlphaStableFlowLinear"]
    model_name = {
        "GaussianFlowLinear": "Lipman et al",
        "AlphaStableFlowLinear": rf"Our approach ($\alpha = {float(cfg['alpha']):.1f}$)",
    }
    metric_order = [
        ("minority_coverage_at_k", "Minority Coverage@k", "up"),
        ("minority_mass_ratio", "Minority Mass Ratio", "up"),
        ("minority_recall", "Minority Recall", "up"),
        ("class_hist_kl_true_to_gen", "Class-Hist KL(true||gen)", "down"),
        ("runtime_sec", "Runtime (s)", "down"),
    ]

    header = "Metric & " + " & ".join(model_name[m] for m in model_order) + r" \\"
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
    ]

    for metric_key, metric_label, direction in metric_order:
        vals = [float(payload["results"][m][metric_key]) for m in model_order]
        best_idx = vals.index(max(vals) if direction == "up" else min(vals))
        cells = []
        for i, v in enumerate(vals):
            s = _fmt(v)
            if i == best_idx:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        arrow = r"$\uparrow$" if direction == "up" else r"$\downarrow$"
        lines.append(f"    {metric_label} {arrow} & " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{CIFAR100 long-tail benchmark in frozen feature space.}",
        r"  \label{tab:cifar100_longtail_bench3}",
        r"\end{table}",
    ]

    out = table_dir / "bench_3_cifar100_longtail_results.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved table artifact: {out}")
