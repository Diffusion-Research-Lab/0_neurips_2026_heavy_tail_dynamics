"""Generate per-dataset benchmark tables and figures from current eval artifacts."""

from __future__ import annotations

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

from labkit.report import PRETTY_RCPARAMS, format_mean_std_latex  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "benchmarks" / "data"
TABLE_ROOT = REPO_ROOT / "benchmarks" / "tables"
FIGURE_ROOT = REPO_ROOT / "benchmarks" / "figures"

PREFERRED_LABELS = ["GF-Linear", "GF-OT", "DDPM-V", "DLPM", "TEDM-Orig"]
DISPLAY_MODEL_LABELS = {
    "gaussian_flow_linear": "GF-Linear",
    "gaussian_flow_ot": "GF-OT",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "tedm_origin": "TEDM-Orig",
}
MODEL_COLORS = {
    "GF-Linear": "tab:blue",
    "GF-OT": "tab:pink",
    "DDPM-V": "tab:green",
    "DLPM": "tab:orange",
    "TEDM-Orig": "tab:olive",
}
LINE_WIDTH = 2.8
MAX_EPOCH_POINTS = 20
DATASET_LABELS = {
    "alpha_stable_target": "Alpha-stable iso.",
    "alpha_stable_mixture_target": "Alpha-stable mix.",
    "kddcup": "KDD Cup",
    "cifar100_lt": "CIFAR100-LT",
    "hrrr": "HRRR",
    "imagenet_lt": "ImageNet-LT",
    "lvis": "LVIS",
}

PERFORMANCE_TABLE_METRICS = ["MMD_RBF", "TAIL_COVERAGE_ERROR", "MSSLE"]
INSPECT_TABLE_METRICS = ["init_error", "training_loss_error"]
CURVE_METRICS = ["MMD_RBF", "TAIL_COVERAGE_ERROR", "MSSLE"]
TRAIN_METRICS = ["training_loss", "training_loss_std", "grad_norm_epoch"]

METRIC_LABELS = {
    "MMD_RBF": "MMD RBF",
    "TAIL_COVERAGE_ERROR": "Tail Cov. Err.",
    "MSSLE": "Quantile Log Err.",
    "init_error": "Init. Error",
    "training_loss_error": "Train Loss Err. (MSE)",
    "training_loss": "Training Loss",
    "training_loss_std": "Training Loss Std.",
    "grad_norm_epoch": "Grad Norm",
}
METRIC_FILENAMES = {
    "MMD_RBF": "mmd_rbf",
    "TAIL_COVERAGE_ERROR": "tail_cov_err",
    "MSSLE": "quantile_log_err",
    "training_loss": "training_loss",
    "training_loss_std": "training_loss_std",
    "grad_norm_epoch": "grad_norm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render benchmark tables and figures from eval artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--table-root", type=Path, default=TABLE_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset presets to render.")
    return parser.parse_args()


def ordered_labels(labels) -> list[str]:
    labels = list(dict.fromkeys(labels))
    return [label for label in PREFERRED_LABELS if label in labels] + sorted(
        label for label in labels if label not in PREFERRED_LABELS
    )


def dataset_title(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name.replace("_", " "))


def dataset_slug(dataset_name: str) -> str:
    aliases = {
        "alpha_stable_target": "alpha_stable_iso",
        "alpha_stable_mixture_target": "alpha_stable_mix",
    }
    return aliases.get(dataset_name, dataset_name)


def discover_eval_batches(artifact_root: Path) -> dict[str, list[Path]]:
    dataset_batches: dict[str, list[Path]] = {}
    for batch_dir in sorted(path for path in artifact_root.glob("*_evaluate") if path.is_dir()):
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


def load_batch_scalars(batch_dir: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        scalars_path = run_dir / "scalars.csv.gz"
        if scalars_path.exists():
            frames.append(pd.read_csv(scalars_path))
    if not frames:
        raise FileNotFoundError(f"No scalars.csv.gz found under {batch_dir}")
    frame = pd.concat(frames, ignore_index=True)
    for column in ["dataset_alpha", "model_alpha", "value", "checkpoint_epoch", "epoch"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_dataset_scalars(batch_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for batch_dir in batch_dirs:
        frame = load_batch_scalars(batch_dir)
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


def best_parameter_rows(summary: pd.DataFrame, metric_name: str, fixed_cols: list[str]) -> pd.DataFrame:
    sub = summary[summary["metric_name"].eq(metric_name)].dropna(subset=["median"]).copy()
    if sub.empty:
        return sub
    rows = []
    for _, group in sub.groupby(fixed_cols, dropna=False):
        if "model_alpha" in group.columns and not group["model_alpha"].isna().all():
            rows.append(group.nsmallest(1, "median").iloc[0].to_dict())
        else:
            rows.append(group.iloc[0].to_dict())
    return pd.DataFrame(rows)


def build_metric_table(summary: pd.DataFrame, metric_names: list[str], row_labels: list[str]) -> pd.DataFrame:
    table = pd.DataFrame("NA", index=row_labels, columns=[METRIC_LABELS[name] for name in metric_names], dtype=object)
    for metric_name in metric_names:
        winners = best_parameter_rows(summary, metric_name, ["model_label"])
        if winners.empty:
            continue
        best_label = winners.loc[winners["median"].idxmin(), "model_label"]
        col_name = METRIC_LABELS[metric_name]
        for _, row in winners.iterrows():
            caption = ""
            if not pd.isna(row.get("model_alpha")):
                caption = f"a={float(row['model_alpha']):.3g}"
            value = float(row["median"])
            if metric_name == "init_error":
                value = max(1e-12, value)
            table.loc[row["model_label"], col_name] = format_mean_std_latex(
                value,
                float(row["std"]),
                caption=caption,
                bold=row["model_label"] == best_label,
            )
    return table


def dataframe_to_latex_table(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    align = "l" + "c" * len(frame.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        "Model & " + " & ".join(frame.columns) + " \\\\",
        "\\midrule",
    ]
    for row_label, row in frame.iterrows():
        lines.append(f"{row_label} & " + " & ".join(str(value) for value in row.values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_metric_curve(
    summary: pd.DataFrame,
    *,
    metric_name: str,
    x_key: str,
    dataset_name: str,
    output_path: Path,
    ylabel: str,
    log_scale: bool = True,
) -> None:
    models = ordered_labels(summary["model_label"].dropna().unique())
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for model_label in models:
        curve = summary[(summary["metric_name"].eq(metric_name)) & (summary["model_label"].eq(model_label))].sort_values(x_key)
        if curve.empty:
            continue
        if x_key == "epoch" and len(curve) > MAX_EPOCH_POINTS:
            sample_idx = np.unique(np.round(np.linspace(0, len(curve) - 1, num=MAX_EPOCH_POINTS)).astype(int))
            curve = curve.iloc[sample_idx]
        ax.plot(
            curve[x_key],
            curve["median"],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=3.5,
            label=model_label,
            alpha=0.8,
            color=MODEL_COLORS.get(model_label, "tab:gray"),
        )
    ax.set_title(f"{dataset_title(dataset_name)} | {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.2, which="both")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(2, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_last_jacobian_curves(batch_dirs: list[Path]) -> tuple[dict[int, dict[str, list[np.ndarray]]], list[str], list[int]]:
    curves_by_epoch: dict[int, dict[str, list[np.ndarray]]] = {}
    model_labels: set[str] = set()
    for batch_dir in batch_dirs:
        for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
            jacobian_path = run_dir / "jacobian_last_epoch.npz"
            scalars_path = run_dir / "scalars.csv.gz"
            if not jacobian_path.exists() or not scalars_path.exists():
                continue
            sample = pd.read_csv(scalars_path, nrows=1)
            if sample.empty:
                continue
            row = sample.iloc[0]
            model_label = DISPLAY_MODEL_LABELS.get(str(row.get("model_name", "")), str(row.get("model_label", "")))
            payload = np.load(jacobian_path)
            checkpoint_epoch = int(np.asarray(payload["checkpoint_epoch"]).item())
            curve = np.asarray(payload["curve"], dtype=float)
            model_labels.add(model_label)
            curves_by_epoch.setdefault(checkpoint_epoch, {}).setdefault(model_label, []).append(curve)
    ordered_models = ordered_labels(model_labels)
    selected_epochs = sorted(curves_by_epoch)
    for checkpoint_epoch in selected_epochs:
        for model_label in ordered_models:
            curves_by_epoch[checkpoint_epoch].setdefault(model_label, [])
    return curves_by_epoch, ordered_models, selected_epochs


def save_jacobian_curve_panels(
    curves_by_epoch: dict[int, dict[str, list[np.ndarray]]],
    *,
    models: list[str],
    checkpoint_epochs: list[int],
    dataset_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_epochs:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Jacobian spectral norm curve\nnot available in current eval artifacts.",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.suptitle(dataset_title(dataset_name), y=0.98)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(checkpoint_epochs), figsize=(2.8 * len(checkpoint_epochs), 2.8), sharex=True, sharey=True)
    axes = [axes] if len(checkpoint_epochs) == 1 else list(axes)
    for ax, checkpoint_epoch in zip(axes, checkpoint_epochs):
        x_values = None
        for model_label in models:
            values_list = curves_by_epoch[checkpoint_epoch][model_label]
            if not values_list:
                continue
            values = np.stack(values_list, axis=0)
            x_values = np.arange(values.shape[1])
            ax.plot(
                x_values,
                np.median(values, axis=0),
                linewidth=LINE_WIDTH,
                label=model_label,
                alpha=0.8,
                color=MODEL_COLORS.get(model_label, "tab:gray"),
            )
        ax.set_title(f"Checkpoint {checkpoint_epoch}")
        ax.set_xlabel("Time trajectory")
        if x_values is not None:
            ax.set_xticks([0, x_values[-1]], ["data", "noise"])
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("Jac. Spectral Norm")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(2, len(labels)), frameon=False)
    fig.suptitle(dataset_title(dataset_name), y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_dataset(
    dataset_name: str,
    batch_dirs: list[Path],
    *,
    table_root: Path,
    figure_root: Path,
) -> None:
    scalars = load_dataset_scalars(batch_dirs)
    row_labels = ordered_labels(scalars["model_label"].dropna().unique())

    perf_summary = summarize_source_metrics(scalars, "test_metrics", ["model_label", "model_alpha"])
    perf_table = build_metric_table(perf_summary, PERFORMANCE_TABLE_METRICS, row_labels)
    perf_tex = dataframe_to_latex_table(
        perf_table,
        caption=f"{dataset_title(dataset_name)} benchmark performance. Lower is better for every metric.",
        label=f"tab:bench-{dataset_slug(dataset_name)}-performance",
    )
    save_text(table_root / f"{dataset_slug(dataset_name)}__performance.tex", perf_tex)

    inspect_summary = summarize_source_metrics(scalars, "inspect", ["model_label", "model_alpha"])
    inspect_table = build_metric_table(inspect_summary, INSPECT_TABLE_METRICS, row_labels)
    inspect_tex = dataframe_to_latex_table(
        inspect_table,
        caption=f"{dataset_title(dataset_name)} inspection metrics. Lower is better for every metric.",
        label=f"tab:bench-{dataset_slug(dataset_name)}-inspect",
    )
    save_text(table_root / f"{dataset_slug(dataset_name)}__inspect.tex", inspect_tex)

    test_evolution = summarize_source_metrics(scalars, "test_metrics", ["checkpoint_epoch", "model_label"])
    for metric_name in CURVE_METRICS:
        save_metric_curve(
            test_evolution,
            metric_name=metric_name,
            x_key="checkpoint_epoch",
            dataset_name=dataset_name,
            output_path=figure_root / f"{dataset_slug(dataset_name)}__{METRIC_FILENAMES[metric_name]}.pdf",
            ylabel=METRIC_LABELS[metric_name],
        )

    train_evolution = summarize_source_metrics(scalars, "train_stats", ["epoch", "model_label"])
    for metric_name in TRAIN_METRICS:
        save_metric_curve(
            train_evolution,
            metric_name=metric_name,
            x_key="epoch",
            dataset_name=dataset_name,
            output_path=figure_root / f"{dataset_slug(dataset_name)}__{METRIC_FILENAMES[metric_name]}.pdf",
            ylabel=METRIC_LABELS[metric_name],
        )

    jacobian_curves, models, checkpoint_epochs = load_last_jacobian_curves(batch_dirs)
    save_jacobian_curve_panels(
        jacobian_curves,
        models=models,
        checkpoint_epochs=checkpoint_epochs,
        dataset_name=dataset_name,
        output_path=figure_root / f"{dataset_slug(dataset_name)}__jacobian_spectral_norm.pdf",
    )


def main() -> int:
    args = parse_args()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
