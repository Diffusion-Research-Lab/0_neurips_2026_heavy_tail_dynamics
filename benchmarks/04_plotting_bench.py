"""Generate per-dataset benchmark tables and figures from current eval artifacts."""

import argparse
from pathlib import Path
import sys
import matplotlib.ticker as mticker
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
    "GF-Linear Euler": "#1f77b4",
    "GF-Linear Heun": "#17becf",
    "DDPM-V DDPM": "#2ca02c",
    "DDPM-V DDIM": "#9467bd",
    "DLPM alpha=1.7": "#ff7f0e",
    "DLPM alpha=1.9": "#d62728",
    "TEDM nu=2.1": "#8c564b",
    "TEDM nu=3.0": "#7f7f7f",
    "GF-Linear": "#1f77b4",
    "DDPM-V": "#2ca02c",
    "DLPM": "#ff7f0e",
    "TEDM-Orig": "#8c564b",
}
FALLBACK_MODEL_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#17becf", "#8c564b", "#7f7f7f"]
MODEL_MARKERS = {
    "GF-Linear Euler": "o",
    "GF-Linear Heun": "s",
    "DDPM-V DDPM": "^",
    "DDPM-V DDIM": "D",
    "DLPM alpha=1.7": "v",
    "DLPM alpha=1.9": "P",
    "TEDM nu=2.1": "X",
    "TEDM nu=3.0": "*",
    "GF-Linear": "o",
    "DDPM-V": "^",
    "DLPM": "v",
    "TEDM-Orig": "X",
}
FALLBACK_MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINE_WIDTH = 2.0
LEGEND_FONTSIZE = 7
TOP_LEGEND_Y = 1.08
TOP_LEGEND_COLUMNS = 3
TOP_LEGEND_LAYOUT_RECT = (0, 0, 1, 0.88)
FULL_LAYOUT_RECT = (0, 0, 1, 1)
MMD_BOXPLOT_MIN_WIDTH = 4.0
MMD_BOXPLOT_BASE_WIDTH = 0.85
MMD_BOXPLOT_MODEL_WIDTH = 0.60
MMD_BOXPLOT_HEIGHT = 3.2
TCE_FIGSIZE = (3.9, 5.0)
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
    "TV_CLASSIFIER",
    "TCE(90)",
    "TCE(99)",
    "TCE(99,9)",
    "TCE(99,99)",
]
TCE_QUANTILE_MIN = 90.0
TCE_QUANTILE_MAX = 99.99
TCE_N_QUANTILES = 20
TCE_ANCHOR_LABELS = {
    90.0: ("TCE(90)", "90", ["TCE(90)", "TCE(90%)"]),
    99.0: ("TCE(99)", "99", ["TCE(99)", "TCE(99%)"]),
    99.9: ("TCE(99,9)", "99.9", ["TCE(99,9)", "TCE(99.9)", "TCE(99.9%)"]),
    99.99: ("TCE(99,99)", "99.99", ["TCE(99,99)", "TCE(99.99)", "TCE(99.99%)"]),
}


def format_tce_metric_name(quantile: float) -> str:
    for anchor, (name, _, _) in TCE_ANCHOR_LABELS.items():
        if np.isclose(float(quantile), anchor):
            return name
    label = f"{float(quantile):.4f}".rstrip("0").rstrip(".")
    return f"TCE({label})"


def tce_quantile_specs() -> list[tuple[str, float, str, list[str]]]:
    probs = np.geomspace(
        1.0 - TCE_QUANTILE_MIN / 100.0,
        1.0 - TCE_QUANTILE_MAX / 100.0,
        TCE_N_QUANTILES,
    )
    for anchor_quantile in TCE_ANCHOR_LABELS:
        anchor_prob = 1.0 - anchor_quantile / 100.0
        index = int(np.argmin(np.abs(np.log(probs) - np.log(anchor_prob))))
        probs[index] = anchor_prob
    specs = []
    for prob in probs:
        quantile = 100.0 * (1.0 - float(prob))
        anchor = next((anchor for anchor in TCE_ANCHOR_LABELS if np.isclose(quantile, anchor)), None)
        if anchor is None:
            name = format_tce_metric_name(quantile)
            tick_label = f"{quantile:.4f}".rstrip("0").rstrip(".")
            aliases = [name, f"TCE({tick_label}%)"]
        else:
            name, tick_label, aliases = TCE_ANCHOR_LABELS[anchor]
        specs.append((name, quantile, tick_label, aliases))
    return specs


TCE_QUANTILE_SPECS = tce_quantile_specs()
TCE_METRICS = [name for name, _, _, _ in TCE_QUANTILE_SPECS]
TCE_QUANTILES = {name: quantile for name, quantile, _, _ in TCE_QUANTILE_SPECS}
TCE_TICK_LABELS = {name: tick_label for name, _, tick_label, _ in TCE_QUANTILE_SPECS}
TCE_ANCHOR_METRICS = [TCE_ANCHOR_LABELS[quantile][0] for quantile in TCE_ANCHOR_LABELS]
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
TRAIN_METRICS = ["training_loss"]
TEST_VS_TEST_SOURCE = "test_vs_test_metrics"
STALE_DATASET_FIGURE_FILENAMES = [
    "mmd_rbf",
    "tv_classifier",
    "tce_90",
    "tce_95",
    "tce_99",
    "tce_999",
    "tce_9999",
    "grad_norm",
]

METRIC_LABELS = {
    "MMD_RBF": "MMD RBF",
    "TV_CLASSIFIER": "Classifier TV lower bound",
    "TCE(90)": "TCE(90)",
    "TCE(99)": "TCE(99)",
    "TCE(99,9)": "TCE(99,9)",
    "TCE(99,99)": "TCE(99,99)",
    "CLASS_RECOVERY_INDEX": "Class Recovery",
    "CLASS_HIST_TV": "Class Hist. TV",
    "training_loss": "Training Loss",
}
METRIC_FILENAMES = {
    "MMD_RBF": "mmd_rbf",
    "TV_CLASSIFIER": "tv_classifier",
    "TCE(90)": "tce_90",
    "TCE(99)": "tce_99",
    "TCE(99,9)": "tce_999",
    "TCE(99,99)": "tce_9999",
    "CLASS_RECOVERY_INDEX": "class_recovery",
    "CLASS_HIST_TV": "class_hist_tv",
    "training_loss": "training_loss",
}


def discover_eval_batches(artifact_root: Path) -> dict[str, list[Path]]:
    dataset_batches: dict[str, list[Path]] = {}
    for batch_dir in sorted(path for path in artifact_root.glob("*_evaluate") if path.is_dir()):
        if "pilot" in batch_dir.name.lower():
            continue
        sample = None
        for scalars_path in sorted(batch_dir.glob("[0-9][0-9][0-9]*/scalars.csv.gz")):
            sample = pd.read_csv(scalars_path, nrows=1)
            if not sample.empty:
                break
        if sample is None or sample.empty:
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


def model_color(model_label: str) -> str:
    label = str(model_label)
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    index = sum(ord(char) for char in label) % len(FALLBACK_MODEL_COLORS)
    return FALLBACK_MODEL_COLORS[index]


def model_marker(model_label: str) -> str:
    label = str(model_label)
    if label in MODEL_MARKERS:
        return MODEL_MARKERS[label]
    index = sum(ord(char) for char in label) % len(FALLBACK_MODEL_MARKERS)
    return FALLBACK_MODEL_MARKERS[index]


def marker_positions(n_points: int, n_markers: int = 5) -> list[int]:
    if n_points <= n_markers:
        return list(range(n_points))
    return sorted(set(np.linspace(0, n_points - 1, n_markers, dtype=int).tolist()))


def ordered_model_labels(labels) -> list[str]:
    labels = list(dict.fromkeys(label for label in labels if pd.notna(label)))
    return [label for label in PREFERRED_LABELS if label in labels] + sorted(
        label for label in labels if label not in PREFERRED_LABELS
    )


def finite_values(frame: pd.DataFrame, column: str = "value", *, positive: bool = False) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    if positive:
        mask &= values > 0.0
    return values[mask]


def unit_interval_data_ylim(groups: list[np.ndarray], baseline_values: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([*groups, baseline_values])
    high = float(np.max(values)) if values.size else 1.0
    upper = min(1.0, max(0.12, high + max(0.02, 0.15 * high)))
    return 0.0, upper


def save_figure(fig, output_path: Path, *, rect=FULL_LAYOUT_RECT) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_top_legend(ax) -> bool:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return False
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, TOP_LEGEND_Y),
        ncol=TOP_LEGEND_COLUMNS,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )
    return True


def plot_model_curve(
    ax,
    x,
    y,
    model_label: str,
    *,
    alpha: float,
    linewidth: float = LINE_WIDTH,
    markersize: float = 3.5,
    markevery=None,
) -> None:
    ax.plot(
        x,
        y,
        linewidth=linewidth,
        marker=model_marker(model_label),
        markevery=markevery,
        markersize=markersize,
        label=model_label,
        alpha=alpha,
        color=model_color(model_label),
    )


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
    row_labels = ordered_model_labels(summary["model_label"].dropna().unique())
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


def cleanup_dataset_outputs(dataset_slug: str, *, table_root: Path, figure_root: Path) -> None:
    (table_root / f"{dataset_slug}__performance.tex").unlink(missing_ok=True)
    for filename_stem in STALE_DATASET_FIGURE_FILENAMES:
        (figure_root / f"{dataset_slug}__{filename_stem}.pdf").unlink(missing_ok=True)


def render_metric_boxplot(
    final_scalars: pd.DataFrame,
    dataset_slug: str,
    figure_root: Path,
    *,
    metric_name: str,
    positive: bool = False,
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
    adaptive_unit_ylim: bool = False,
) -> None:
    test_metric_mask = final_scalars["source"].eq("test_metrics")
    metric_mask = final_scalars["metric_name"].eq(metric_name)
    metric_rows = final_scalars[test_metric_mask & metric_mask]
    groups = []
    kept_labels = []
    for model_label in ordered_model_labels(metric_rows["model_label"].dropna().unique()):
        values = finite_values(metric_rows[metric_rows["model_label"].eq(model_label)], positive=positive)
        if values.size == 0:
            continue
        groups.append(values)
        kept_labels.append(model_label)

    if not groups:
        return

    fig_width = max(MMD_BOXPLOT_MIN_WIDTH, MMD_BOXPLOT_BASE_WIDTH + MMD_BOXPLOT_MODEL_WIDTH * len(kept_labels))
    fig, ax = plt.subplots(figsize=(fig_width, MMD_BOXPLOT_HEIGHT))
    box = ax.boxplot(
        groups,
        widths=0.42,
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
        patch.set_facecolor(model_color(model_label))
        patch.set_alpha(0.28)
        patch.set_label(model_label)
    ax.set_xticks(range(1, len(kept_labels) + 1), kept_labels)

    baseline_mask = final_scalars["source"].eq(TEST_VS_TEST_SOURCE)
    baseline_values = finite_values(final_scalars[baseline_mask & metric_mask], positive=positive)
    if baseline_values.size:
        ax.axhline(float(np.median(baseline_values)), color="#777777", linestyle=":", linewidth=1.8, label="test-vs-test")

    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif adaptive_unit_ylim:
        ax.set_ylim(*unit_interval_data_ylim(groups, baseline_values))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.2, axis="y", which="both")
    add_top_legend(ax)
    filename = METRIC_FILENAMES.get(metric_name, metric_name.lower())
    save_figure(fig, figure_root / f"{dataset_slug}__{filename}_boxplot.pdf", rect=TOP_LEGEND_LAYOUT_RECT)


def render_tce_quantiles(final_scalars: pd.DataFrame, dataset_slug: str, figure_root: Path) -> None:
    tce_scalars = final_scalars.copy()
    tce_scalars["metric_name"] = tce_scalars["metric_name"].map(lambda name: TCE_METRIC_ALIASES.get(name, name))
    summary = summarize_source_metrics(tce_scalars, "test_metrics", ["model_label"])
    summary = summary[summary["metric_name"].isin(TCE_METRICS)].copy()
    baseline_summary = summarize_source_metrics(tce_scalars, TEST_VS_TEST_SOURCE, [])
    baseline_summary = baseline_summary[baseline_summary["metric_name"].isin(TCE_METRICS)].set_index("metric_name")
    if summary.empty and baseline_summary.empty:
        return

    fig, ax = plt.subplots(figsize=TCE_FIGSIZE)
    baseline_points = []
    for metric_name in TCE_METRICS:
        if metric_name not in baseline_summary.index:
            continue
        mean = float(baseline_summary.loc[metric_name, "mean"])
        if np.isfinite(mean) and mean > 0.0:
            baseline_points.append((TCE_QUANTILES[metric_name] / 100.0, mean))
    if baseline_points:
        xs, means = zip(*baseline_points)
        ax.plot(
            xs,
            means,
            color="#777777",
            linestyle=":",
            linewidth=1.8,
            marker=".",
            markersize=3.0,
            markevery=marker_positions(len(baseline_points)),
            label="test-vs-test",
        )

    for model_label in ordered_model_labels(summary["model_label"].dropna().unique()):
        rows = summary[summary["model_label"].eq(model_label)].set_index("metric_name")
        points = []
        for metric_name in TCE_METRICS:
            if metric_name not in rows.index:
                continue
            mean = float(rows.loc[metric_name, "mean"])
            if np.isfinite(mean):
                points.append((TCE_QUANTILES[metric_name] / 100.0, mean))
        if points:
            xs, means = zip(*points)
            plot_model_curve(
                ax,
                xs,
                means,
                model_label,
                linewidth=1.8,
                markersize=3.0,
                markevery=marker_positions(len(points)),
                alpha=0.64,
            )

    if not add_top_legend(ax):
        plt.close(fig)
        return

    ax.set_xlabel("Tail quantile (%)")
    ax.set_ylabel("TCE, upper tail log error")
    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    x_ticks = [TCE_QUANTILES[name] / 100.0 for name in TCE_ANCHOR_METRICS]
    x_tick_labels = [TCE_TICK_LABELS[name] for name in TCE_ANCHOR_METRICS]
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(x_tick_labels))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(TCE_QUANTILE_MIN / 100.0, TCE_QUANTILE_MAX / 100.0)
    ax.grid(alpha=0.2, which="both")
    save_figure(fig, figure_root / f"{dataset_slug}__tce_quantiles.pdf", rect=TOP_LEGEND_LAYOUT_RECT)


def render_training_curves(scalars: pd.DataFrame, dataset_slug: str, figure_root: Path) -> None:
    train_evolution = summarize_source_metrics(scalars, "train_stats", ["epoch", "model_label"])
    if train_evolution.empty:
        return

    for metric_name in TRAIN_METRICS:
        metric_rows = train_evolution[train_evolution["metric_name"].eq(metric_name)]
        if metric_rows.empty:
            continue

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for model_label in ordered_model_labels(metric_rows["model_label"].dropna().unique()):
            curve = metric_rows[metric_rows["model_label"].eq(model_label)].sort_values("epoch")
            curve = curve[pd.to_numeric(curve["epoch"], errors="coerce") > 0]
            if curve.empty:
                continue
            plot_model_curve(
                ax,
                curve["epoch"],
                curve["median"],
                model_label,
                markevery=marker_positions(len(curve)),
                alpha=0.7,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS[metric_name])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")
        if not add_top_legend(ax):
            plt.close(fig)
            continue
        save_figure(fig, figure_root / f"{dataset_slug}__{METRIC_FILENAMES[metric_name]}.pdf", rect=TOP_LEGEND_LAYOUT_RECT)


def render_dataset(
    dataset_name: str,
    batch_dirs: list[Path],
    *,
    table_root: Path,
    figure_root: Path,
) -> None:
    scalars = load_dataset_scalars(batch_dirs)
    dataset_slug = DATASET_SLUG_ALIASES.get(dataset_name, dataset_name)
    cleanup_dataset_outputs(dataset_slug, table_root=table_root, figure_root=figure_root)

    final_scalars = final_checkpoint_rows(scalars)
    render_metric_boxplot(final_scalars, dataset_slug, figure_root, metric_name="MMD_RBF", positive=True, yscale="log")
    render_metric_boxplot(final_scalars, dataset_slug, figure_root, metric_name="TV_CLASSIFIER", adaptive_unit_ylim=True)
    render_tce_quantiles(final_scalars, dataset_slug, figure_root)
    render_training_curves(scalars, dataset_slug, figure_root)


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
