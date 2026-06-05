#!/usr/bin/env python
"""Analyze pilot evaluation results and generate final benchmark configs."""

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

import matplotlib                                                                                    # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                      # noqa
import numpy as np                                                                                   # noqa
import pandas as pd                                                                                  # noqa
import yaml                                                                                          # noqa
from benchmarks.utils import load_yaml                                                               # noqa

ARTIFACT_ROOT = PROJECT_ROOT / "benchmarks" / "artifacts"
CONFIG_ROOT = PROJECT_ROOT / "benchmarks" / "configs"
REPORT_ROOT = PROJECT_ROOT / "benchmarks" / "reports"
BENCH_CONFIG_ROOT = CONFIG_ROOT / "bench"

PILOT_SELECTION_METRICS = ["INNER_LOSS_VAL", "INNER_LOSS_TRAIN", "INNER_LOSS_TEST"]
METRIC_LABELS = {
    "INNER_LOSS_VAL": "Validation inner loss",
    "INNER_LOSS_TRAIN": "Training inner loss",
    "INNER_LOSS_TEST": "Test inner loss",
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
    "cifar100_lt": "cifar100_lt",
    "imagenet_lt": "imagenet_lt",
    "hrrr": "hrrr",
    "lvis": "lvis",
}
FAMILY_SPECS = {
    "synth": {
        "pilot_config": CONFIG_ROOT / "pilot" / "synth.yaml",
        "bench_template": CONFIG_ROOT / "templates" / "synth_bench.yaml",
        "batch_pattern": "*_evaluate",
        "match_terms": ("synth", "alphastable"),
    },
    "image": {
        "pilot_config": CONFIG_ROOT / "pilot" / "image.yaml",
        "bench_template": CONFIG_ROOT / "templates" / "image_bench.yaml",
        "batch_pattern": "*_evaluate",
        "match_terms": ("image",),
    },
}
DATASET_TRAIN_OVERRIDES = {}


def discover_family_batch(artifact_root: Path, family: str) -> Path:
    spec = FAMILY_SPECS[family]
    candidates = [
        path for path in artifact_root.glob(spec["batch_pattern"])
        if path.is_dir() and "pilot" in path.name.lower() and any(term in path.name.lower() for term in spec["match_terms"])
    ]
    if not candidates:
        raise FileNotFoundError(f"No evaluated pilot batch found for family={family!r} under {artifact_root}.")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def first_non_null(frame: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    if column not in frame.columns or frame.empty:
        return default
    value = frame[column].iloc[0]
    return default if pd.isna(value) else value


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT, help="Root directory containing benchmark artifacts.")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT, help="Output directory for CSV/Markdown/PNG reports.")
    parser.add_argument("--bench-config-root", type=Path, default=BENCH_CONFIG_ROOT, help="Output directory for generated benchmark configs.")
    parser.add_argument("--skip-reports", action="store_true", help="Do not write CSV/Markdown/PNG reports.")
    parser.add_argument("--skip-configs", action="store_true", help="Do not write benchmark configs.")
    args = parser.parse_args()
    frames: list[pd.DataFrame] = []
    for family in FAMILY_SPECS:
        batch_dir = discover_family_batch(args.artifact_root, family)
        for artifact_run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
            scalars_path = artifact_run_dir / "scalars.csv.gz"
            if not scalars_path.exists():
                continue

            summary_path = artifact_run_dir / "summary.yaml"
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            summary = summary or {}
            frame = pd.read_csv(scalars_path)
            frame = frame[frame["source"].isin(["pilot_selection", "test_metrics"])].copy()
            if frame.empty:
                continue

            frame["checkpoint_epoch"] = pd.to_numeric(frame["checkpoint_epoch"], errors="coerce")
            final_epoch = int(frame["checkpoint_epoch"].dropna().max())
            frame = frame[frame["checkpoint_epoch"].eq(final_epoch)].copy()

            run_tokens = artifact_run_dir.name.split("__")
            dataset_preset = first_non_null(frame, "dataset_preset", run_tokens[1] if len(run_tokens) > 1 else "dataset")
            dataset_name = first_non_null(frame, "dataset_name", dataset_preset)
            model_preset = first_non_null(frame, "model_preset", run_tokens[3] if len(run_tokens) > 3 else "model")
            train_preset = first_non_null(frame, "train_preset", run_tokens[4] if len(run_tokens) > 4 else "train")
            inferred_model_name = str(model_preset)
            for ordered_model_name in MODEL_ORDER:
                if inferred_model_name.startswith(ordered_model_name):
                    inferred_model_name = ordered_model_name
                    break
            model_name = first_non_null(frame, "model_name", inferred_model_name)
            token = str(train_preset).removeprefix("pilot_lr")
            match = re.fullmatch(r"(\d+)e(\d+)", token)
            if match:
                compact_lr = float(f"{match.group(1)}e-{match.group(2)}")
            else:
                try:
                    compact_lr = float("nan") if token is None else float(token)
                except (TypeError, ValueError):
                    compact_lr = float("nan")
            train_lr_value = first_non_null(frame, "train_lr", compact_lr)
            try:
                train_lr = float("nan") if train_lr_value is None else float(train_lr_value)
            except (TypeError, ValueError):
                train_lr = float("nan")
            setting_parts = [str(model_preset), str(train_preset)]
            if pd.notna(train_lr):
                setting_parts.append(f"lr={train_lr:.3g}")

            batch_name = artifact_run_dir.parent.name.lower()
            if "synth" in batch_name or "alphastable" in batch_name:
                bench_name = "synth"
            elif "image" in batch_name:
                bench_name = "image"
            else:
                raise ValueError(f"Could not infer family from batch dir: {artifact_run_dir.parent}")

            frame["artifact_batch_dir"] = str(artifact_run_dir.parent)
            frame["artifact_run_dir"] = str(artifact_run_dir)
            frame["raw_run_dir"] = summary.get("source_run_dir")
            frame["bench_name"] = bench_name
            frame["dataset_preset"] = dataset_preset
            frame["dataset_name"] = dataset_name
            frame["dataset_slug"] = DATASET_SLUGS.get(str(dataset_preset), str(dataset_preset))
            frame["model_name"] = model_name
            frame["model_label"] = first_non_null(frame, "model_label", MODEL_LABELS.get(model_name, model_name))
            frame["model_preset"] = model_preset
            frame["train_preset"] = train_preset
            frame["train_lr"] = train_lr
            frame["setting_key"] = f"{model_preset}__{train_preset}"
            frame["setting_label"] = " | ".join(setting_parts)
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No evaluated pilot runs with scalars.csv.gz found under {args.artifact_root}.")

    pilot_frame = pd.concat(frames, ignore_index=True)
    pilot_frame["value"] = pd.to_numeric(pilot_frame["value"], errors="coerce")
    pilot_frame["eval_repeat_idx"] = pd.to_numeric(pilot_frame.get("eval_repeat_idx"), errors="coerce")

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
    setting_summary = (
        pilot_frame.groupby(group_cols, dropna=False)["value"]
        .agg(mean="mean", median="median", std=lambda series: float(series.std(ddof=0)), n="size")
        .reset_index()
    )
    setting_summary["std"] = setting_summary["std"].fillna(0.0)

    recommendations: list[dict[str, Any]] = []
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
        available_metrics = [metric for metric in PILOT_SELECTION_METRICS if metric in score_matrix.columns and score_matrix[metric].notna().any()]
        if not available_metrics:
            continue
        selection_metric = available_metrics[0]
        ranked = meta.merge(
            pd.DataFrame(
                {
                    "setting_key": score_matrix.index,
                    "selection_metric": selection_metric,
                    "selection_score": score_matrix[selection_metric].values,
                }
            ),
            on="setting_key",
            how="inner",
        )
        ranked = ranked.sort_values(["selection_score", "setting_label"], kind="stable").reset_index(drop=True)

        best = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        reason = f"lowest {METRIC_LABELS.get(selection_metric, selection_metric).lower()} ({best['selection_score']:.6g})"
        if runner_up is not None:
            reason += f"; next best={runner_up['selection_score']:.6g}"

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
                "selection_metric": selection_metric,
                "selection_score": float(best["selection_score"]),
                "why": reason,
                "artifact_batch_dir": group["artifact_batch_dir"].iloc[0] if "artifact_batch_dir" in group.columns else None,
            }
        )

    recommendations_df = pd.DataFrame(recommendations)
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values(["bench_name", "dataset_slug", "model_name"]).reset_index(drop=True)
    if recommendations_df.empty:
        raise RuntimeError("No usable pilot recommendations were produced.")

    if not args.skip_reports:
        args.report_root.mkdir(parents=True, exist_ok=True)
        csv_path = args.report_root / "pilot_best_settings.csv"
        md_path = args.report_root / "pilot_best_settings.md"
        png_path = args.report_root / "pilot_summary.png"

        recommendations_df.to_csv(csv_path, index=False)
        table_columns = [
            "bench_name",
            "dataset_slug",
            "model_name",
            "selected_model_preset",
            "selected_pilot_train_preset",
            "selected_lr",
            "selection_metric",
            "selection_score",
        ]
        display = recommendations_df[table_columns].copy()
        display["selected_lr"] = display["selected_lr"].map(lambda value: f"{value:.4g}")
        display["selection_score"] = display["selection_score"].map(lambda value: f"{value:.6g}")
        header = "| " + " | ".join(table_columns) + " |"
        separator = "| " + " | ".join(["---"] * len(table_columns)) + " |"
        rows = ["| " + " | ".join(str(row[col]) for col in table_columns) + " |" for _, row in display.iterrows()]
        md_path.write_text("# Pilot Best Settings\n\n" + "\n".join([header, separator, *rows]) + "\n", encoding="utf-8")

        families = ["synth", "image"]
        fig, axes = plt.subplots(len(families), 1, figsize=(15, 10), constrained_layout=True)
        axes = [axes] if len(families) == 1 else list(axes)
        for ax, family in zip(axes, families):
            family_frame = recommendations_df[recommendations_df["bench_name"].eq(family)].copy()
            datasets = sorted(family_frame["dataset_slug"].unique().tolist())
            models = [model for model in MODEL_ORDER if model in family_frame["model_name"].unique()]
            score_matrix = np.full((len(datasets), len(models)), np.nan)

            for row in family_frame.itertuples(index=False):
                y_idx = datasets.index(row.dataset_slug)
                x_idx = models.index(row.model_name)
                score_matrix[y_idx, x_idx] = row.selection_score

            image = ax.imshow(score_matrix, aspect="auto", cmap="viridis_r")
            ax.set_xticks(range(len(models)), [MODEL_LABELS.get(model, model) for model in models], rotation=20, ha="right")
            ax.set_yticks(range(len(datasets)), datasets)
            ax.set_title(f"{family} pilot winners")

            for row in family_frame.itertuples(index=False):
                y_idx = datasets.index(row.dataset_slug)
                x_idx = models.index(row.model_name)
                annotation = f"{row.selected_model_preset}\n{row.selected_pilot_train_preset}"
                ax.text(x_idx, y_idx, annotation, ha="center", va="center", fontsize=8, color="black")

            fig.colorbar(
                image,
                ax=ax,
                fraction=0.025,
                pad=0.01,
                label="Selected validation inner loss",
            )

        fig.savefig(png_path, dpi=200)
        plt.close(fig)
    if not args.skip_configs:
        if args.bench_config_root.exists():
            for path in sorted(args.bench_config_root.rglob("*.yaml"), reverse=True):
                path.unlink()
            for path in sorted((path for path in args.bench_config_root.rglob("*") if path.is_dir()), reverse=True):
                if path != args.bench_config_root:
                    path.rmdir()

        args.bench_config_root.mkdir(parents=True, exist_ok=True)
        for _, row in recommendations_df.iterrows():
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
            final_train_name = "selected"
            final_train_cfg = copy.deepcopy(bench_template["trains"][bench_template["sweep"]["trains"][0]])
            final_train_cfg["lr"] = train_lr
            final_train_cfg.update(DATASET_TRAIN_OVERRIDES.get(dataset_preset, {}))

            for section_name, entry_name in (("datasets", dataset_preset), ("networks", network_name)):
                if entry_name not in bench_template[section_name]:
                    raise KeyError(f"Missing {section_name}.{entry_name} in template config.")

            config = {
                "run": {
                    **copy.deepcopy(bench_template["run"]),
                    "name": f"bench_{family}__{dataset_slug}__{model_name}",
                },
                "selection": {
                    "dataset_slug": dataset_slug,
                    "selected_model_preset": model_preset,
                    "selected_pilot_train_preset": train_preset,
                    "selected_lr": train_lr,
                    "selection_metric": str(row["selection_metric"]),
                    "selection_score": float(row["selection_score"]),
                    "source_artifact_batch_dir": str(row["artifact_batch_dir"]),
                },
                "sweep": {
                    "datasets": [dataset_preset],
                    "networks": [network_name],
                    "models": [model_name],
                    "trains": [final_train_name],
                },
                "datasets": {
                    dataset_preset: copy.deepcopy(bench_template["datasets"][dataset_preset]),
                },
                "networks": {
                    network_name: copy.deepcopy(bench_template["networks"][network_name]),
                },
                "models": {
                    model_name: copy.deepcopy(pilot_config["models"][model_preset]),
                },
                "trains": {
                    final_train_name: final_train_cfg,
                },
                "save": copy.deepcopy(bench_template["save"]),
            }
            output_path = args.bench_config_root / family / dataset_slug / f"{model_name}.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)

    print(f"Loaded {pilot_frame['artifact_run_dir'].nunique()} evaluated pilot runs.")
    print(f"Wrote {len(recommendations_df)} per-dataset/model selections.")
