"""Utilities for benchmark exploitation notebooks."""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.runner._utils import build_dataset, build_model, build_network
from genkit.metrics import metric_on_quantile, mssle, wasserstein_distance


MODEL_LABELS = {
    "gaussian_flow_linear": "GF-Linear",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "dlpm_eps_origin": "DLPM-Orig",
    "flow_matching_origin": "FM-Orig",
    "score_sde_origin": "ScoreSDE-Orig",
}
TRAIN_TAGS = {"short": "S.train", "long": "L.train"}
NETWORK_TAGS = {"mlp_small": "S.nn", "mlp_medium": "L.nn"}
METRIC_XIS = {"MSSLE_95": 0.95, "MSSLE_99": 0.99}
METRIC_ORDER = ["W1", "MSSLE_95", "MSSLE_99"]
RUN_GROUP_COLS = ["run_dir", "checkpoint_epoch", "trial_idx", "dataset_alpha", "model_name", "model_label", "model_alpha", "network_preset", "train_preset"]
GROUP_COLS = ["checkpoint_epoch", "dataset_alpha", "model_name", "model_label", "model_alpha", "network_preset", "train_preset"]


def run_dirs(root: Path) -> list[Path]:
    """Return benchmark run directories sorted by numeric prefix."""
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name[:3].isdigit())


def available_checkpoint_epochs(run_dir: Path) -> list[int]:
    """Return available retained checkpoint epochs for one run."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    epochs = []
    for path in ckpt_dir.glob("ckpt_epoch_*.pt"):
        try:
            epochs.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(epochs))


def resolve_checkpoint(run_dir: Path, checkpoint_epoch: int | None = None) -> tuple[Path, int | None]:
    """Resolve which checkpoint file to use for one run."""
    if checkpoint_epoch is None:
        return run_dir / "checkpoint.pt", None

    ckpt_path = run_dir / "checkpoints" / f"ckpt_epoch_{int(checkpoint_epoch):04d}.pt"
    if not ckpt_path.exists():
        available = available_checkpoint_epochs(run_dir)
        raise FileNotFoundError(
            f"Checkpoint epoch {checkpoint_epoch} not found in {run_dir}. Available epochs: {available}"
        )
    return ckpt_path, int(checkpoint_epoch)


def pretty_model(name: str) -> str:
    """Map an internal model name to a short display label."""
    return MODEL_LABELS.get(name, name)


def setting_code(row: pd.Series) -> str:
    """Return the compact train/network/alpha code for one summary row."""
    pieces = [TRAIN_TAGS[row["train_preset"]], NETWORK_TAGS[row["network_preset"]]]
    if "checkpoint_epoch" in row and not pd.isna(row["checkpoint_epoch"]):
        pieces.append(f"e{int(row['checkpoint_epoch'])}")
    if not pd.isna(row["model_alpha"]):
        pieces.append(f"a{float(row['model_alpha']):.3g}")
    return "(" + "/".join(pieces) + ")"


def _metric_fields(x_ref: torch.Tensor, x_gen: torch.Tensor) -> dict[str, float]:
    """Compute the benchmark metrics for one reference/generated pair."""
    row = {
        "W1": float(wasserstein_distance(x_ref, x_gen, p=1)),
    }
    row.update(
        {
            metric_name: float(metric_on_quantile(mssle, x_ref, x_gen, xi=xi, reduction="mean"))
            for metric_name, xi in METRIC_XIS.items()
        }
    )
    return row


def load_generator(
    run_dir: Path,
    *,
    device: str = "cpu",
    checkpoint_epoch: int | None = None,
) -> tuple[Any, torch.Tensor, dict[str, Any], int | None]:
    """Reload one trained generator and its test split from a run directory."""
    checkpoint_path, resolved_epoch = resolve_checkpoint(run_dir, checkpoint_epoch=checkpoint_epoch)
    final_checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dtype_name = str(final_checkpoint["model_init"]["dtype"]).split(".")[-1]
    dtype = getattr(torch, dtype_name)
    network_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["network"]))
    model_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["model"]))
    model_cfg.setdefault("params", {})
    model_cfg["params"]["fdtype"] = dtype
    model_cfg["params"]["device"] = device

    x_train, _, x_test = build_dataset(config["dataset"], dtype=dtype, device=device)
    net, _ = build_network(network_cfg, x_train)
    net = net.to(device=device, dtype=dtype)
    generator, _ = build_model(model_cfg, net, x_train, dtype=dtype, device=device)
    generator._net.load_state_dict(checkpoint["network_state_dict"] if "network_state_dict" in checkpoint else checkpoint["model_state"])
    generator._net.eval()
    return generator, x_test, config, resolved_epoch


def load_train_stats(run_dir: Path, *, checkpoint_epoch: int | None = None) -> pd.DataFrame:
    """Load per-epoch training statistics for one benchmark run."""
    frame = pd.read_csv(run_dir / "train_stats.csv")
    if checkpoint_epoch is not None and "epoch" in frame.columns:
        frame = frame[frame["epoch"] <= int(checkpoint_epoch)].copy()
    return frame


def evaluate_run(
    run_dir: Path,
    *,
    n_samples: int,
    n_repeats: int,
    device: str = "cpu",
    checkpoint_epoch: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one saved run several times and return one row per repeat."""
    generator, x_test, config, resolved_epoch = load_generator(run_dir, device=device, checkpoint_epoch=checkpoint_epoch)
    x_ref = x_test[: min(n_samples, len(x_test)), :]
    base_row = {
        "run_dir": run_dir.name,
        "checkpoint_epoch": resolved_epoch if resolved_epoch is not None else int(config["train"]["n_epochs"]),
        "trial_idx": int(config.get("run", {}).get("trial_idx", 0)),
        "dataset_alpha": float(config["dataset"]["params"]["alpha"]),
        "model_name": config["model"]["name"],
        "model_label": pretty_model(config["model"]["name"]),
        "model_alpha": config["model"]["params"].get("alpha", np.nan),
        "network_preset": config["network"]["preset_name"],
        "train_preset": config["train"]["preset_name"],
    }

    rows = []
    for repeat_idx in range(n_repeats):
        x_gen = generator.sample(n_samples=len(x_ref))
        rows.append({**base_row, "repeat_idx": repeat_idx, **_metric_fields(x_ref, x_gen)})
    return rows


def summarize_exact(repeat_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeat-level metrics for each exact benchmark setting."""
    rows = []
    for keys, group in repeat_df.groupby(GROUP_COLS, dropna=False):
        row = dict(zip(GROUP_COLS, keys, strict=True))
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(repeat_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeat-level metrics for each saved benchmark run."""
    rows = []
    for keys, group in repeat_df.groupby(RUN_GROUP_COLS, dropna=False):
        row = dict(zip(RUN_GROUP_COLS, keys, strict=True))
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def reduce_best_alpha(summary_exact: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Reduce alpha-parametrized models by best alpha at fixed data/train/network."""
    groups = ["checkpoint_epoch", "dataset_alpha", "model_label", "network_preset", "train_preset"]
    rows = []
    for _, group in summary_exact.groupby(groups, dropna=False):
        best = group.iloc[0] if group["model_alpha"].isna().all() else group.nsmallest(1, f"{metric}_mean").iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def best_table_df(summary_exact: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pick the best setting for each dataset-alpha/model pair."""
    groups = ["checkpoint_epoch", "dataset_alpha", "model_label"]
    rows = [group.nsmallest(1, f"{metric}_mean").iloc[0].to_dict() for _, group in summary_exact.groupby(groups, dropna=False)]
    return pd.DataFrame(rows)


def aggregated_best_alpha(summary_exact: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average each condition over dataset alpha after best-alpha reduction."""
    reduced = reduce_best_alpha(summary_exact, metric)
    rows = []
    for keys, group in reduced.groupby(["checkpoint_epoch", "model_label", "network_preset", "train_preset"], dropna=False):
        row = dict(zip(["checkpoint_epoch", "model_label", "network_preset", "train_preset"], keys, strict=True))
        row[metric] = float(group[f"{metric}_mean"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def reduce_best_alpha_runs(run_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Reduce alpha-parametrized runs by best alpha at fixed data/train/network/trial."""
    groups = ["checkpoint_epoch", "dataset_alpha", "model_label", "network_preset", "train_preset", "trial_idx"]
    rows = []
    for _, group in run_summary.groupby(groups, dropna=False):
        best = group.iloc[0] if group["model_alpha"].isna().all() else group.nsmallest(1, f"{metric}_mean").iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def grouped_violin(ax, series_by_label, title, ylabel, color="#4C72B0"):
    """Draw a compact violin plot with overlaid sample points."""
    labels = list(series_by_label)
    values = [np.asarray(series_by_label[label], dtype=float) for label in labels]
    pos = np.arange(1, len(labels) + 1)
    violins = ax.violinplot(values, positions=pos, widths=0.8, showmeans=True, showextrema=False)
    for body in violins["bodies"]:
        body.set_alpha(0.28)
        body.set_facecolor(color)
        body.set_edgecolor(color)
    violins["cmeans"].set_color("black")
    for idx, vals in enumerate(values, start=1):
        jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=12, alpha=0.7, color="black")
    ax.set_xticks(pos, labels)
    ax.tick_params(axis="x", labelrotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, axis="y")


def styled_table(ax, cell_text, row_labels, col_labels, *, fontsize=8.5, scale=(1.0, 1.35), corner_text=None):
    """Render a compact styled matplotlib table."""
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(*scale)
    if corner_text is not None:
        corner_width = table[(1, -1)].get_width() if (1, -1) in table.get_celld() else table[(1, 0)].get_width()
        corner_height = table[(0, 0)].get_height()
        table.add_cell(0, -1, width=corner_width, height=corner_height, text=corner_text, loc="center")
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.2")
        cell.set_linewidth(0.6)
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")
    return table


def average_curve(curves):
    """Interpolate a list of variable-length epoch curves onto a common grid."""
    if not curves:
        return None, None, None
    max_epoch = max(total_epochs for total_epochs, _ in curves)
    epoch_grid = np.arange(1, max_epoch + 1, dtype=float)
    aligned = []
    for total_epochs, curve in curves:
        x_curve = np.linspace(1.0, float(total_epochs), num=len(curve), dtype=float)
        aligned.append(np.interp(epoch_grid, x_curve, curve, left=np.nan, right=np.nan))
    stack = np.stack(aligned, axis=0)
    return epoch_grid, np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
