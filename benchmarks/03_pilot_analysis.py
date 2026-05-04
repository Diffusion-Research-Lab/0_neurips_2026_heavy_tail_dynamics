#!/usr/bin/env python
"""Analyze pilot evaluation results and generate final benchmark configs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from benchmarks.utils import load_yaml  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "benchmarks" / "data"
CONFIG_ROOT = PROJECT_ROOT / "benchmarks" / "configs"
REPORT_ROOT = PROJECT_ROOT / "benchmarks" / "reports"
BENCH_CONFIG_ROOT = CONFIG_ROOT / "bench"

PERF_METRICS = ["FID", "MMD_RBF", "SLICED_WASSERSTEIN", "TAIL_COVERAGE_ERROR", "MSSLE"]
METRIC_LABELS = {
    "FID": "FID",
    "MMD_RBF": "MMD RBF",
    "SLICED_WASSERSTEIN": "Sliced Wasserstein",
    "TAIL_COVERAGE_ERROR": "Tail coverage error",
    "MSSLE": "MSSLE",
}
MODEL_ORDER = ["gaussian_flow_ot", "gaussian_flow_linear", "ddpm_v", "dlpm_eps", "tedm_origin"]
MODEL_LABELS = {
    "gaussian_flow_ot": "GF-OT",
    "gaussian_flow_linear": "GF-Linear",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "tedm_origin": "TEDM-Orig",
}
DATASET_SLUGS = {
    "alpha_stable_target": "alpha_stable_iso",
    "alpha_stable_mixture_target": "alpha_stable_mix",
    "kddcup": "kddcup",
    "cifar100_lt": "cifar100_lt",
    "imagenet_lt": "imagenet_lt",
    "hrrr": "hrrr",
    "lvis": "lvis",
}
FAMILY_SPECS = {
    "synth": {
        "pilot_config": CONFIG_ROOT / "pilot" / "synth.yaml",
        "bench_template": CONFIG_ROOT / "templates" / "synth_bench.yaml",
        "batch_pattern": "*_pilot_evaluate",
        "match": "alphastable",
    },
    "real": {
        "pilot_config": CONFIG_ROOT / "pilot" / "real.yaml",
        "bench_template": CONFIG_ROOT / "templates" / "real_bench.yaml",
        "batch_pattern": "*_pilot_evaluate",
        "match": "real",
    },
    "image": {
        "pilot_config": CONFIG_ROOT / "pilot" / "image.yaml",
        "bench_template": CONFIG_ROOT / "templates" / "image_bench.yaml",
        "batch_pattern": "*_pilot_evaluate",
        "match": "image",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Root directory containing evaluated pilot batches.")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT, help="Output directory for CSV/Markdown/PNG reports.")
    parser.add_argument("--bench-config-root", type=Path, default=BENCH_CONFIG_ROOT, help="Output directory for generated benchmark configs.")
    parser.add_argument("--skip-reports", action="store_true", help="Do not write CSV/Markdown/PNG reports.")
    parser.add_argument("--skip-configs", action="store_true", help="Do not write benchmark configs.")
    return parser.parse_args()


def discover_family_batch(data_root: Path, family: str) -> Path:
    spec = FAMILY_SPECS[family]
    candidates = [
        path for path in data_root.glob(spec["batch_pattern"])
        if path.is_dir() and spec["match"] in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No evaluated pilot batch found for family={family!r} under {data_root}.")
    return sorted(candidates)[-1]


def safe_read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def maybe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def infer_bench_name(batch_dir: Path) -> str:
    name = batch_dir.name.lower()
    if "alphastable" in name:
        return "synth"
    if "real" in name:
        return "real"
    if "image" in name:
        return "image"
    raise ValueError(f"Could not infer family from batch dir: {batch_dir}")


def first_non_null(frame: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    if column not in frame.columns or frame.empty:
        return default
    value = frame[column].iloc[0]
    return default if pd.isna(value) else value


def parse_compact_lr(train_preset: str) -> float:
    token = str(train_preset).removeprefix("pilot_lr")
    match = re.fullmatch(r"(\d+)e(\d+)", token)
    if not match:
        return maybe_float(token)
    base, exponent = match.groups()
    return float(f"{base}e-{exponent}")


def infer_model_name(model_preset: str) -> str:
    model_preset = str(model_preset)
    for model_name in MODEL_ORDER:
        if model_preset.startswith(model_name):
            return model_name
    return model_preset


def build_setting_label(model_preset: str, train_preset: str, train_lr: float) -> str:
    parts = [model_preset, train_preset]
    if pd.notna(train_lr):
        parts.append(f"lr={train_lr:.3g}")
    return " | ".join(parts)


def extract_run_metadata(frame: pd.DataFrame, artifact_run_dir: Path) -> dict[str, Any]:
    run_tokens = artifact_run_dir.name.split("__")
    dataset_preset = first_non_null(frame, "dataset_preset", run_tokens[1] if len(run_tokens) > 1 else "dataset")
    dataset_name = first_non_null(frame, "dataset_name", dataset_preset)
    model_preset = first_non_null(frame, "model_preset", run_tokens[3] if len(run_tokens) > 3 else "model")
    train_preset = first_non_null(frame, "train_preset", run_tokens[4] if len(run_tokens) > 4 else "train")
    model_name = first_non_null(frame, "model_name", infer_model_name(model_preset))
    train_lr = maybe_float(first_non_null(frame, "train_lr", parse_compact_lr(train_preset)))

    return {
        "bench_name": infer_bench_name(artifact_run_dir.parent),
        "dataset_preset": dataset_preset,
        "dataset_name": dataset_name,
        "dataset_slug": DATASET_SLUGS.get(str(dataset_preset), str(dataset_preset)),
        "model_name": model_name,
        "model_label": first_non_null(frame, "model_label", MODEL_LABELS.get(model_name, model_name)),
        "model_preset": model_preset,
        "train_preset": train_preset,
        "train_lr": train_lr,
        "setting_key": f"{model_preset}__{train_preset}",
        "setting_label": build_setting_label(str(model_preset), str(train_preset), train_lr),
    }


def load_artifact_run(artifact_run_dir: Path) -> pd.DataFrame | None:
    scalars_path = artifact_run_dir / "scalars.csv.gz"
    if not scalars_path.exists():
        return None

    summary = safe_read_yaml(artifact_run_dir / "summary.yaml") or {}
    frame = pd.read_csv(scalars_path)
    frame = frame[frame["source"].eq("test_metrics")].copy()
    if frame.empty:
        return None

    frame["checkpoint_epoch"] = pd.to_numeric(frame["checkpoint_epoch"], errors="coerce")
    final_epoch = int(frame["checkpoint_epoch"].dropna().max())
    frame = frame[frame["checkpoint_epoch"].eq(final_epoch)].copy()

    meta = extract_run_metadata(frame, artifact_run_dir)
    frame["artifact_batch_dir"] = str(artifact_run_dir.parent)
    frame["artifact_run_dir"] = str(artifact_run_dir)
    frame["raw_run_dir"] = summary.get("source_run_dir")
    for key, value in meta.items():
        frame[key] = value
    return frame


def load_batches(data_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in FAMILY_SPECS:
        batch_dir = discover_family_batch(data_root, family)
        for artifact_run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
            loaded = load_artifact_run(artifact_run_dir)
            if loaded is not None:
                frames.append(loaded)
    if not frames:
        raise FileNotFoundError(f"No evaluated pilot runs with scalars.csv.gz found under {data_root}.")
    frame = pd.concat(frames, ignore_index=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["eval_repeat_idx"] = pd.to_numeric(frame.get("eval_repeat_idx"), errors="coerce")
    return frame


def summarize_settings(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "bench_name",
        "dataset_name",
        "dataset_preset",
        "dataset_slug",
        "artifact_batch_dir",
        "model_name",
        "model_label",
        "model_preset",
        "train_preset",
        "train_lr",
        "setting_key",
        "setting_label",
        "metric_name",
    ]
    summary = (
        frame.groupby(group_cols, dropna=False)["value"]
        .agg(mean="mean", median="median", std=lambda s: float(s.std(ddof=0)), n="size")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def rank_settings(setting_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    recommendations: list[dict[str, Any]] = []
    pivot_tables: dict[tuple[str, str, str], pd.DataFrame] = {}

    meta_cols = [
        "setting_key",
        "setting_label",
        "model_label",
        "model_preset",
        "train_preset",
        "train_lr",
    ]

    for (bench_name, dataset_name, model_name), group in setting_summary.groupby(["bench_name", "dataset_name", "model_name"], dropna=False):
        meta = group[meta_cols].drop_duplicates(subset=["setting_key"]).reset_index(drop=True)
        score_matrix = group.pivot_table(index="setting_key", columns="metric_name", values="mean", aggfunc="first")
        available_metrics = [metric for metric in PERF_METRICS if metric in score_matrix.columns and score_matrix[metric].notna().any()]
        if not available_metrics:
            continue
        score_matrix = score_matrix[available_metrics]

        rank_matrix = score_matrix.rank(axis=0, method="min", ascending=True)
        ranked = meta.merge(
            pd.DataFrame(
                {
                    "setting_key": score_matrix.index,
                    "avg_rank": rank_matrix.mean(axis=1).values,
                    "n_metric_wins": (rank_matrix == 1).sum(axis=1).values,
                    "n_metrics": score_matrix.notna().sum(axis=1).values,
                    "win_labels": [
                        ", ".join(METRIC_LABELS.get(metric_name, metric_name) for metric_name in available_metrics if rank_matrix.loc[key, metric_name] == 1)
                        for key in score_matrix.index
                    ],
                }
            ),
            on="setting_key",
            how="inner",
        )
        ranked = ranked.sort_values(["avg_rank", "setting_label"], kind="stable").reset_index(drop=True)

        best = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        reason = f"best average rank ({best['avg_rank']:.2f}) across {len(available_metrics)} metrics"
        if best["win_labels"]:
            reason += f"; wins: {best['win_labels']}"
        if runner_up is not None:
            reason += f"; next best avg rank={runner_up['avg_rank']:.2f}"

        recommendations.append(
            {
                "bench_name": bench_name,
                "dataset_name": dataset_name,
                "dataset_preset": group["dataset_preset"].iloc[0],
                "dataset_slug": group["dataset_slug"].iloc[0],
                "model_name": model_name,
                "model_label": best["model_label"],
                "selected_model_preset": best["model_preset"],
                "selected_pilot_train_preset": best["train_preset"],
                "selected_lr": best["train_lr"],
                "keep_setting": best["setting_label"],
                "avg_rank": best["avg_rank"],
                "n_metric_wins": int(best["n_metric_wins"]),
                "n_metrics": int(best["n_metrics"]),
                "available_metrics": ", ".join(METRIC_LABELS[m] for m in available_metrics),
                "why": reason,
                "artifact_batch_dir": group["artifact_batch_dir"].iloc[0] if "artifact_batch_dir" in group.columns else None,
            }
        )
        pivot_tables[(bench_name, dataset_name, model_name)] = ranked

    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        return recommendations_df, pivot_tables
    recommendations_df = recommendations_df.sort_values(["bench_name", "dataset_slug", "model_name"]).reset_index(drop=True)
    return recommendations_df, pivot_tables


def render_markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "bench_name",
        "dataset_slug",
        "model_name",
        "selected_model_preset",
        "selected_pilot_train_preset",
        "selected_lr",
        "avg_rank",
        "n_metric_wins",
    ]
    display = frame[columns].copy()
    display["selected_lr"] = display["selected_lr"].map(lambda value: f"{value:.4g}")
    display["avg_rank"] = display["avg_rank"].map(lambda value: f"{value:.2f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, separator, *rows]) + "\n"


def plot_summary(frame: pd.DataFrame, output_path: Path) -> None:
    families = ["synth", "real", "image"]
    fig, axes = plt.subplots(len(families), 1, figsize=(15, 10), constrained_layout=True)
    axes = [axes] if len(families) == 1 else list(axes)

    for ax, family in zip(axes, families):
        family_frame = frame[frame["bench_name"].eq(family)].copy()
        datasets = sorted(family_frame["dataset_slug"].unique().tolist())
        models = [model for model in MODEL_ORDER if model in family_frame["model_name"].unique()]
        rank_matrix = np.full((len(datasets), len(models)), np.nan)

        for row in family_frame.itertuples(index=False):
            y_idx = datasets.index(row.dataset_slug)
            x_idx = models.index(row.model_name)
            rank_matrix[y_idx, x_idx] = row.avg_rank

        image = ax.imshow(rank_matrix, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(models)), [MODEL_LABELS.get(model, model) for model in models], rotation=20, ha="right")
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set_title(f"{family} pilot winners")

        for row in family_frame.itertuples(index=False):
            y_idx = datasets.index(row.dataset_slug)
            x_idx = models.index(row.model_name)
            annotation = f"{row.selected_model_preset}\n{row.selected_pilot_train_preset}"
            ax.text(x_idx, y_idx, annotation, ha="center", va="center", fontsize=8, color="black")

        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Average rank")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def clone_single_entry(config: dict[str, Any], section_name: str, entry_name: str) -> dict[str, Any]:
    section = copy.deepcopy(config[section_name])
    if entry_name not in section:
        raise KeyError(f"Missing {section_name}.{entry_name} in template config.")
    return {entry_name: section[entry_name]}


def selected_train_name(lr: float) -> str:
    return "selected"


def build_bench_config(row: pd.Series) -> dict[str, Any]:
    family = str(row["bench_name"])
    family_spec = FAMILY_SPECS[family]
    pilot_config = load_yaml(family_spec["pilot_config"])
    bench_template = load_yaml(family_spec["bench_template"])

    dataset_preset = str(row["dataset_preset"])
    dataset_slug = str(row["dataset_slug"])
    model_name = str(row["model_name"])
    model_preset = str(row["selected_model_preset"])
    train_preset = str(row["selected_pilot_train_preset"])
    train_lr = float(row["selected_lr"])

    network_name = str(bench_template["sweep"]["networks"][0])
    final_train_name = selected_train_name(train_lr)
    final_train_cfg = copy.deepcopy(bench_template["trains"][bench_template["sweep"]["trains"][0]])
    final_train_cfg["lr"] = train_lr

    run_name = f"bench_{family}__{dataset_slug}__{model_name}"
    return {
        "run": {
            **copy.deepcopy(bench_template["run"]),
            "name": run_name,
        },
        "selection": {
            "dataset_slug": dataset_slug,
            "selected_model_preset": model_preset,
            "selected_pilot_train_preset": train_preset,
            "selected_lr": train_lr,
            "selection_metric": "average_rank",
            "avg_rank": float(row["avg_rank"]),
            "n_metric_wins": int(row["n_metric_wins"]),
            "available_metrics": str(row["available_metrics"]),
            "source_artifact_batch_dir": str(row["artifact_batch_dir"]),
        },
        "sweep": {
            "datasets": [dataset_preset],
            "networks": [network_name],
            "models": [model_name],
            "trains": [final_train_name],
        },
        "datasets": clone_single_entry(bench_template, "datasets", dataset_preset),
        "networks": clone_single_entry(bench_template, "networks", network_name),
        "models": {
            model_name: copy.deepcopy(pilot_config["models"][model_preset]),
        },
        "trains": {
            final_train_name: final_train_cfg,
        },
        "save": copy.deepcopy(bench_template["save"]),
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_reports(recommendations_df: pd.DataFrame, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "pilot_best_settings.csv"
    md_path = report_root / "pilot_best_settings.md"
    png_path = report_root / "pilot_summary.png"

    recommendations_df.to_csv(csv_path, index=False)
    md_contents = "# Pilot Best Settings\n\n" + render_markdown_table(recommendations_df)
    md_path.write_text(md_contents, encoding="utf-8")
    plot_summary(recommendations_df, png_path)


def clear_existing_bench_configs(bench_config_root: Path) -> None:
    if not bench_config_root.exists():
        return
    for path in sorted(bench_config_root.rglob("*.yaml"), reverse=True):
        path.unlink()
    for path in sorted((path for path in bench_config_root.rglob("*") if path.is_dir()), reverse=True):
        if path != bench_config_root:
            path.rmdir()


def write_bench_configs(recommendations_df: pd.DataFrame, bench_config_root: Path) -> None:
    clear_existing_bench_configs(bench_config_root)
    bench_config_root.mkdir(parents=True, exist_ok=True)
    for _, row in recommendations_df.iterrows():
        family = str(row["bench_name"])
        dataset_slug = str(row["dataset_slug"])
        model_name = str(row["model_name"])
        config = build_bench_config(row)
        output_path = bench_config_root / family / dataset_slug / f"{model_name}.yaml"
        write_yaml(output_path, config)


def main() -> int:
    args = parse_args()
    pilot_frame = load_batches(args.data_root)
    setting_summary = summarize_settings(pilot_frame)
    recommendations_df, _ = rank_settings(setting_summary)
    if recommendations_df.empty:
        raise RuntimeError("No usable pilot recommendations were produced.")

    if not args.skip_reports:
        write_reports(recommendations_df, args.report_root)
    if not args.skip_configs:
        write_bench_configs(recommendations_df, args.bench_config_root)

    print(f"Loaded {pilot_frame['artifact_run_dir'].nunique()} evaluated pilot runs.")
    print(f"Wrote {len(recommendations_df)} per-dataset/model selections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
