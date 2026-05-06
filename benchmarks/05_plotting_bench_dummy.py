"""Generate per-dataset dummy benchmark tables from current eval artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from labkit.report import format_mean_std_latex  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "benchmarks" / "data"
TABLE_ROOT = REPO_ROOT / "benchmarks" / "tables_dummy"

PREFERRED_LABELS = ["GF-Linear", "GF-OT", "DDPM", "DLPM", "t-EDM"]
DISPLAY_MODEL_LABELS = {
    "gaussian_flow_linear": "GF-Linear",
    "gaussian_flow_ot": "GF-OT",
    "ddpm_v": "DDPM",
    "dlpm_eps": "DLPM",
    "tedm_origin": "t-EDM",
    "DDPM-V": "DDPM",
    "TEDM-Orig": "t-EDM",
}
DATASET_LABELS = {
    "alpha_stable_target": "Alpha-stable iso.",
    "alpha_stable_mixture_target": "Alpha-stable mix.",
    "kddcup": "KDD Cup",
    "cifar100_lt": "CIFAR100-LT",
    "hrrr": "HRRR",
    "imagenet_lt": "ImageNet-LT",
    "lvis": "LVIS",
}
DEFAULT_DATASETS = list(DATASET_LABELS)
PERFORMANCE_TABLE_METRICS = [
    "MMD_RBF",
    "TCE(90%)",
    "TCE(95%)",
    "TCE(99%)",
    "TCE(99.9%)",
]
METRIC_LABELS = {
    "MMD_RBF": "MMD RBF $\\downarrow$",
    "TCE(90%)": "TCE(90\\%) $\\downarrow$",
    "TCE(95%)": "TCE(95\\%) $\\downarrow$",
    "TCE(99%)": "TCE(99\\%) $\\downarrow$",
    "TCE(99.9%)": "TCE(99.9\\%) $\\downarrow$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dummy benchmark tables from eval artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--table-root", type=Path, default=TABLE_ROOT)
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
    return dataset_batches


def load_batch_scalars(batch_dir: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        scalars_path = run_dir / "scalars.csv.gz"
        if scalars_path.exists():
            frames.append(pd.read_csv(scalars_path))
    if not frames:
        raise FileNotFoundError(f"No scalars.csv.gz found under {batch_dir}")
    return pd.concat(frames, ignore_index=True)


def load_dataset_scalars(batch_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for batch_dir in batch_dirs:
        frame = load_batch_scalars(batch_dir)
        frame["eval_batch_dir"] = batch_dir.name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["model_label"] = frame["model_label"].replace(DISPLAY_MODEL_LABELS)
    frame["model_name"] = frame["model_name"].replace(DISPLAY_MODEL_LABELS)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def summarize_source_metrics(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    summary = (
        frame[frame["source"].eq(source)]
        .groupby(["model_label", "metric_name"], dropna=False)["value"]
        .agg(median="median", std=lambda s: float(s.std(ddof=0)))
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def build_metric_table(summary: pd.DataFrame, metric_names: list[str], row_labels: list[str]) -> pd.DataFrame:
    table = pd.DataFrame("NA", index=row_labels, columns=[METRIC_LABELS[name] for name in metric_names], dtype=object)
    for metric_name in metric_names:
        rows = summary[summary["metric_name"].eq(metric_name)].dropna(subset=["median"]).copy()
        if rows.empty:
            continue
        best_label = rows.loc[rows["median"].idxmin(), "model_label"]
        col_name = METRIC_LABELS[metric_name]
        for _, row in rows.iterrows():
            table.loc[row["model_label"], col_name] = format_mean_std_latex(
                float(row["median"]),
                float(row["std"]),
                bold=row["model_label"] == best_label,
            )
    return table


def build_na_table(metric_names: list[str], row_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        "NA",
        index=row_labels,
        columns=[METRIC_LABELS[name] for name in metric_names],
        dtype=object,
    )


def dataframe_to_latex_table(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    align = "l" + "c" * len(frame.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
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


def render_dataset(dataset_name: str, batch_dirs: list[Path], *, table_root: Path) -> None:
    scalars = load_dataset_scalars(batch_dirs)
    row_labels = ordered_labels(scalars["model_label"].dropna().unique())
    perf_summary = summarize_source_metrics(scalars, "test_metrics")
    perf_table = build_metric_table(perf_summary, PERFORMANCE_TABLE_METRICS, row_labels)
    perf_tex = dataframe_to_latex_table(
        perf_table,
        caption=f"\\textbf{{{dataset_title(dataset_name)} dataset.}}",
        label=f"tab:bench-{dataset_slug(dataset_name)}-performance-dummy",
    )
    save_text(table_root / f"{dataset_slug(dataset_name)}__performance.tex", perf_tex)


def render_missing_dataset(dataset_name: str, *, table_root: Path) -> None:
    perf_table = build_na_table(PERFORMANCE_TABLE_METRICS, PREFERRED_LABELS)
    perf_tex = dataframe_to_latex_table(
        perf_table,
        caption=f"\\textbf{{{dataset_title(dataset_name)} dataset.}}",
        label=f"tab:bench-{dataset_slug(dataset_name)}-performance-dummy",
    )
    save_text(table_root / f"{dataset_slug(dataset_name)}__performance.tex", perf_tex)


def main() -> int:
    args = parse_args()
    dataset_batches = discover_eval_batches(args.artifact_root)
    selected_datasets = list(args.datasets) if args.datasets else DEFAULT_DATASETS

    for dataset_name in selected_datasets:
        batch_dirs = dataset_batches.get(dataset_name)
        if not batch_dirs:
            render_missing_dataset(dataset_name, table_root=args.table_root)
            print(f"[done] dataset={dataset_name} tables={args.table_root} (filled with NA)")
            continue
        render_dataset(dataset_name, batch_dirs, table_root=args.table_root)
        print(f"[done] dataset={dataset_name} tables={args.table_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
