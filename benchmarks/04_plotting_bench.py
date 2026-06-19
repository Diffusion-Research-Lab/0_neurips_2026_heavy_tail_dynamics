"""Generate per-dataset benchmark tables and figures from current eval artifacts."""

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from labkit.report import PRETTY_RCPARAMS, format_mean_std_latex                                      # noqa

ARTIFACT_ROOT = REPO_ROOT / "benchmarks" / "artifacts"
TABLE_ROOT = REPO_ROOT / "benchmarks" / "tables"
FIGURE_ROOT = REPO_ROOT / "benchmarks" / "figures"

PREFERRED_LABELS = [
    "GF-Linear Euler",
    "GF-Linear Heun",
    "DDPM-V DDPM",
    "DDPM-V DDIM",
    "DLPM alpha=1.7",
    "DLPM alpha=1.9",
    "TEDM nu=2.1",
    "TEDM nu=3.0",
]
DISPLAY_MODEL_LABELS = {
    "gaussian_flow_linear_euler": "GF-Linear Euler",
    "gaussian_flow_linear_heun": "GF-Linear Heun",
    "gaussian_flow_linear": "GF-Linear",
    "ddpm_v_ddpm": "DDPM-V DDPM",
    "ddpm_v_ddim": "DDPM-V DDIM",
    "dlpm_eps_a17": "DLPM alpha=1.7",
    "dlpm_eps_a19": "DLPM alpha=1.9",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "tedm_origin_nu21": "TEDM nu=2.1",
    "tedm_origin_nu30": "TEDM nu=3.0",
    "tedm_origin": "TEDM-Orig",
}
MODEL_COLORS = {
    "GF-Linear Euler": "tab:blue",
    "GF-Linear Heun": "tab:cyan",
    "DDPM-V DDPM": "tab:green",
    "DDPM-V DDIM": "tab:purple",
    "DLPM alpha=1.7": "tab:orange",
    "DLPM alpha=1.9": "tab:red",
    "TEDM nu=2.1": "tab:olive",
    "TEDM nu=3.0": "tab:brown",
    "GF-Linear": "tab:blue",
    "DDPM-V": "tab:green",
    "DLPM": "tab:orange",
    "TEDM-Orig": "tab:olive",
}
LINE_WIDTH = 2.8
DATASET_LABELS = {
    "alpha_stable_target": "Alpha-stable iso.",
    "alpha_stable_mixture_target": "Alpha-stable mix.",
    "cifar100_lt": "CIFAR100-LT",
    "hrrr": "HRRR",
    "imagenet_lt": "ImageNet-LT",
    "lvis": "LVIS",
}
DATASET_SLUG_ALIASES = {
    "alpha_stable_target": "alpha_stable_iso",
    "alpha_stable_mixture_target": "alpha_stable_mix",
}

PERFORMANCE_TABLE_METRICS = [
    "MMD_RBF",
    "TCE(90)",
    "TCE(99)",
    "TCE(99,9)",
    "TCE(99,99)",
]
TCE_QUANTILE_SPECS = [
    ("TCE(90)", 90.0, "90", ["TCE(90)", "TCE(90%)"]),
    ("TCE(99)", 99.0, "99", ["TCE(99)", "TCE(99%)"]),
    ("TCE(99,9)", 99.9, "99,9", ["TCE(99,9)", "TCE(99.9)", "TCE(99.9%)"]),
    ("TCE(99,99)", 99.99, "99,99", ["TCE(99,99)", "TCE(99.99)", "TCE(99.99%)"]),
]
TCE_METRICS = [name for name, _, _, _ in TCE_QUANTILE_SPECS]
TCE_QUANTILES = {name: quantile for name, quantile, _, _ in TCE_QUANTILE_SPECS}
TCE_TICK_LABELS = {name: tick_label for name, _, tick_label, _ in TCE_QUANTILE_SPECS}
TCE_METRIC_ALIASES = {
    alias: name
    for name, _, _, aliases in TCE_QUANTILE_SPECS
    for alias in aliases
}
CLASS_RECOVERY_DATASETS = ["cifar100_lt", "imagenet_lt"]
CLASS_RECOVERY_METRICS = [
    ("CLASS_RECOVERY_INDEX", True),
    ("CLASS_HIST_TV", False),
]
TRAIN_METRICS = ["training_loss", "grad_norm"]
TEST_VS_TEST_SOURCE = "test_vs_test_metrics"
STALE_DATASET_FIGURE_FILENAMES = [
    "mmd_rbf",
    "tce_90",
    "tce_95",
    "tce_99",
    "tce_999",
    "tce_9999",
]

METRIC_LABELS = {
    "MMD_RBF": "MMD RBF",
    "TCE(90)": "TCE(90)",
    "TCE(99)": "TCE(99)",
    "TCE(99,9)": "TCE(99,9)",
    "TCE(99,99)": "TCE(99,99)",
    "CLASS_RECOVERY_INDEX": "Class Recovery",
    "CLASS_HIST_TV": "Class Hist. TV",
    "training_loss": "Training Loss",
    "grad_norm": "Grad Norm",
}
METRIC_FILENAMES = {
    "MMD_RBF": "mmd_rbf",
    "TCE(90)": "tce_90",
    "TCE(99)": "tce_99",
    "TCE(99,9)": "tce_999",
    "TCE(99,99)": "tce_9999",
    "CLASS_RECOVERY_INDEX": "class_recovery",
    "CLASS_HIST_TV": "class_hist_tv",
    "training_loss": "training_loss",
    "grad_norm": "grad_norm",
}


def discover_eval_batches(artifact_root: Path) -> dict[str, list[Path]]:
    dataset_batches: dict[str, list[Path]] = {}
    for batch_dir in sorted(path for path in artifact_root.glob("*_evaluate") if path.is_dir()):
        if "pilot" in batch_dir.name.lower():
            continue
        run_dirs = sorted(path for path in batch_dir.iterdir() if path.is_dir())
        if not run_dirs:
            continue
        scalars_path = run_dirs[0] / "scalars.csv.gz"
        if not scalars_path.exists():
            continue
        sample = pd.read_csv(scalars_path, nrows=1)
        if sample.empty:
            continue
        dataset_name = str(sample.iloc[0].get("dataset_preset", sample.iloc[0].get("dataset_name", "")))
        if not dataset_name:
            continue
        dataset_batches.setdefault(dataset_name, []).append(batch_dir)
    if not dataset_batches:
        raise FileNotFoundError(f"No evaluation batches found under {artifact_root}")
    return dataset_batches


def load_dataset_scalars(batch_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for batch_dir in batch_dirs:
        batch_frames = []
        for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
            scalars_path = run_dir / "scalars.csv.gz"
            if scalars_path.exists():
                batch_frames.append(pd.read_csv(scalars_path))
        if not batch_frames:
            raise FileNotFoundError(f"No scalars.csv.gz found under {batch_dir}")
        frame = pd.concat(batch_frames, ignore_index=True)
        for column in ["dataset_alpha", "model_alpha", "value", "checkpoint_epoch", "epoch"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["eval_batch_dir"] = batch_dir.name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["model_label"] = frame["model_label"].replace(DISPLAY_MODEL_LABELS)
    return frame


def summarize_source_metrics(frame: pd.DataFrame, source: str, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        frame[frame["source"].eq(source)]
        .groupby(group_cols + ["metric_name"], dropna=False)["value"]
        .agg(mean="mean", median="median", std=lambda s: float(s.std(ddof=0)))
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def final_checkpoint_rows(scalars: pd.DataFrame) -> pd.DataFrame:
    if "run_dir" not in scalars.columns or "checkpoint_epoch" not in scalars.columns:
        return scalars
    checkpoint_epoch = pd.to_numeric(scalars["checkpoint_epoch"], errors="coerce")
    group_cols = ["run_dir"]
    if "eval_batch_dir" in scalars.columns:
        group_cols.insert(0, "eval_batch_dir")
    max_epoch = checkpoint_epoch.groupby([scalars[col] for col in group_cols]).transform("max")
    return scalars[checkpoint_epoch.eq(max_epoch)].copy()


def render_image_class_recovery_table(dataset_batches: dict[str, list[Path]], table_root: Path) -> None:
    summaries = []
    for dataset_name in CLASS_RECOVERY_DATASETS:
        batch_dirs = dataset_batches.get(dataset_name)
        if not batch_dirs:
            continue
        scalars = final_checkpoint_rows(load_dataset_scalars(batch_dirs))
        summary = summarize_source_metrics(
            scalars,
            "test_metrics",
            ["dataset_preset", "model_label", "model_alpha"],
        )
        metric_names = [metric_name for metric_name, _ in CLASS_RECOVERY_METRICS]
        summary = summary[summary["metric_name"].isin(metric_names)].copy()
        if not summary.empty:
            summaries.append(summary)

    if not summaries:
        return

    summary = pd.concat(summaries, ignore_index=True)
    row_labels = list(dict.fromkeys(summary["model_label"].dropna().unique()))
    row_labels = [label for label in PREFERRED_LABELS if label in row_labels] + sorted(
        label for label in row_labels if label not in PREFERRED_LABELS
    )
    col_names = [
        f"{DATASET_LABELS.get(dataset_name, dataset_name.replace('_', ' '))} {METRIC_LABELS[metric_name]}"
        for dataset_name in CLASS_RECOVERY_DATASETS
        if dataset_name in set(summary["dataset_preset"])
        for metric_name, _ in CLASS_RECOVERY_METRICS
    ]
    table = pd.DataFrame("NA", index=row_labels, columns=col_names, dtype=object)

    for dataset_name in CLASS_RECOVERY_DATASETS:
        dataset_summary = summary[summary["dataset_preset"].eq(dataset_name)]
        if dataset_summary.empty:
            continue
        for metric_name, higher_is_better in CLASS_RECOVERY_METRICS:
            sub = dataset_summary[dataset_summary["metric_name"].eq(metric_name)].dropna(subset=["median"]).copy()
            if sub.empty:
                winners = sub
            else:
                rows = []
                for _, group in sub.groupby(["model_label"], dropna=False):
                    if "model_alpha" in group.columns and not group["model_alpha"].isna().all():
                        selected = group.nlargest(1, "median") if higher_is_better else group.nsmallest(1, "median")
                        rows.append(selected.iloc[0].to_dict())
                    else:
                        rows.append(group.iloc[0].to_dict())
                winners = pd.DataFrame(rows)
            if winners.empty:
                continue
            best_idx = winners["median"].idxmax() if higher_is_better else winners["median"].idxmin()
            best_label = winners.loc[best_idx, "model_label"]
            col_name = f"{DATASET_LABELS.get(dataset_name, dataset_name.replace('_', ' '))} {METRIC_LABELS[metric_name]}"
            for _, row in winners.iterrows():
                caption = ""
                if not pd.isna(row.get("model_alpha")):
                    caption = f"a={float(row['model_alpha']):.3g}"
                table.loc[row["model_label"], col_name] = format_mean_std_latex(
                    float(row["median"]),
                    float(row["std"]),
                    caption=caption,
                    bold=row["model_label"] == best_label,
                )

    align = "l" + "c" * len(table.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Image class recovery and class-histogram total variation on labeled image benchmarks. Higher is better for recovery; lower is better for TV.}",
        "\\label{tab:bench-image-class-recovery}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        "Model & " + " & ".join(table.columns) + " \\\\",
        "\\midrule",
    ]
    for row_label, row in table.iterrows():
        lines.append(f"{row_label} & " + " & ".join(str(value) for value in row.values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    table_path = table_root / "image_class_recovery.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines), encoding="utf-8")


def render_dataset(
    dataset_name: str,
    batch_dirs: list[Path],
    *,
    table_root: Path,
    figure_root: Path,
) -> None:
    scalars = load_dataset_scalars(batch_dirs)
    dataset_slug = DATASET_SLUG_ALIASES.get(dataset_name, dataset_name)
    dataset_label = DATASET_LABELS.get(dataset_name, dataset_name.replace("_", " "))
    stale_performance_table = table_root / f"{dataset_slug}__performance.tex"
    stale_performance_table.unlink(missing_ok=True)
    for filename_stem in STALE_DATASET_FIGURE_FILENAMES:
        (figure_root / f"{dataset_slug}__{filename_stem}.pdf").unlink(missing_ok=True)

    final_scalars = final_checkpoint_rows(scalars)
    test_mask = final_scalars["source"].eq("test_metrics")
    metric_mask = final_scalars["metric_name"].eq("MMD_RBF")
    metric_rows = final_scalars[test_mask & metric_mask]
    labels = list(dict.fromkeys(metric_rows["model_label"].dropna().unique()))
    labels = [label for label in PREFERRED_LABELS if label in labels] + sorted(
        label for label in labels if label not in PREFERRED_LABELS
    )
    groups = []
    kept_labels = []
    for model_label in labels:
        values = pd.to_numeric(metric_rows[metric_rows["model_label"].eq(model_label)]["value"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        groups.append(values)
        kept_labels.append(model_label)
    if groups:
        fig_width = max(4.8, 1.0 + 0.75 * len(kept_labels))
        fig, ax = plt.subplots(figsize=(fig_width, 3.2))
        box = ax.boxplot(
            groups,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markersize": 3.5, "markerfacecolor": "white", "markeredgecolor": "#333333"},
            medianprops={"color": "#222222", "linewidth": 1.4},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.45},
        )
        for patch, model_label in zip(box["boxes"], kept_labels):
            patch.set_facecolor(MODEL_COLORS.get(model_label, "tab:gray"))
            patch.set_alpha(0.35)
        ax.set_xticks(range(1, len(kept_labels) + 1), kept_labels)

        baseline = None
        if "source" in final_scalars.columns:
            baseline_mask = final_scalars["source"].eq(TEST_VS_TEST_SOURCE)
            baseline_metric_mask = final_scalars["metric_name"].eq("MMD_RBF")
            baseline_values = pd.to_numeric(final_scalars[baseline_mask & baseline_metric_mask]["value"], errors="coerce").to_numpy(dtype=float)
            baseline_values = baseline_values[np.isfinite(baseline_values)]
            if baseline_values.size:
                baseline = float(np.median(baseline_values))
        if baseline is not None:
            ax.axhline(
                baseline,
                color="#777777",
                linestyle=":",
                linewidth=1.8,
                label="test-vs-test",
            )
            ax.legend(loc="best")

        ax.set_title(f"{dataset_label} | MMD RBF")
        ax.set_ylabel("MMD RBF")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        output_path = figure_root / f"{dataset_slug}__mmd_rbf_boxplot.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    tce_scalars = final_scalars
    if "metric_name" in tce_scalars.columns:
        tce_scalars = tce_scalars.copy()
        tce_scalars["metric_name"] = tce_scalars["metric_name"].map(lambda name: TCE_METRIC_ALIASES.get(name, name))
    summary = summarize_source_metrics(tce_scalars, "test_metrics", ["model_label"])
    summary = summary[summary["metric_name"].isin(TCE_METRICS)].copy()
    if not summary.empty:
        models = list(dict.fromkeys(summary["model_label"].dropna().unique()))
        models = [label for label in PREFERRED_LABELS if label in models] + sorted(
            label for label in models if label not in PREFERRED_LABELS
        )
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for model_label in models:
            rows = summary[summary["model_label"].eq(model_label)].set_index("metric_name")
            xs = []
            means = []
            lowers = []
            uppers = []
            for metric_name in TCE_METRICS:
                if metric_name not in rows.index:
                    continue
                row = rows.loc[metric_name]
                mean = float(row["mean"])
                std = float(row["std"])
                if not np.isfinite(mean):
                    continue
                xs.append(TCE_QUANTILES[metric_name])
                means.append(mean)
                lowers.append(max(0.0, mean - std) if np.isfinite(std) else mean)
                uppers.append(mean + std if np.isfinite(std) else mean)
            if not xs:
                continue

            color = MODEL_COLORS.get(model_label, "tab:gray")
            ax.fill_between(xs, lowers, uppers, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                xs,
                means,
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=3.5,
                label=model_label,
                color=color,
                alpha=0.85,
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.set_title(f"{dataset_label} | Tail coverage error")
            ax.set_xlabel("Tail quantile (%)")
            ax.set_ylabel("TCE")
            ax.set_xticks([TCE_QUANTILES[name] for name in TCE_METRICS])
            ax.set_xticklabels([TCE_TICK_LABELS[name] for name in TCE_METRICS])
            ax.grid(alpha=0.2, which="both")
            fig.legend(handles, labels, loc="upper center", ncol=min(2, len(labels)), frameon=False)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            output_path = figure_root / f"{dataset_slug}__tce_quantiles.pdf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    train_evolution = summarize_source_metrics(scalars, "train_stats", ["epoch", "model_label"])
    for metric_name in TRAIN_METRICS:
        models = list(dict.fromkeys(train_evolution["model_label"].dropna().unique()))
        models = [label for label in PREFERRED_LABELS if label in models] + sorted(
            label for label in models if label not in PREFERRED_LABELS
        )
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for model_label in models:
            train_metric_mask = train_evolution["metric_name"].eq(metric_name)
            train_model_mask = train_evolution["model_label"].eq(model_label)
            curve = train_evolution[train_metric_mask & train_model_mask].sort_values("epoch")
            if curve.empty:
                continue
            ax.plot(
                curve["epoch"],
                curve["median"],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=3.5,
                label=model_label,
                alpha=0.8,
                color=MODEL_COLORS.get(model_label, "tab:gray"),
            )
        ax.set_title(f"{dataset_label} | {METRIC_LABELS[metric_name]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS[metric_name])
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(2, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        output_path = figure_root / f"{dataset_slug}__{METRIC_FILENAMES[metric_name]}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Render benchmark tables and figures from eval artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--table-root", type=Path, default=TABLE_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset presets to render.")
    args = parser.parse_args()
    plt.rcParams.update(PRETTY_RCPARAMS)
    dataset_batches = discover_eval_batches(args.artifact_root)
    selected_datasets = list(args.datasets) if args.datasets else sorted(dataset_batches)

    for dataset_name in selected_datasets:
        batch_dirs = dataset_batches.get(dataset_name)
        if not batch_dirs:
            print(f"[skip] dataset={dataset_name!r} not present under {args.artifact_root}")
            continue
        render_dataset(
            dataset_name,
            batch_dirs,
            table_root=args.table_root,
            figure_root=args.figure_root,
        )
        print(f"[done] dataset={dataset_name} tables={args.table_root} figures={args.figure_root}")
    render_image_class_recovery_table(dataset_batches, args.table_root)
