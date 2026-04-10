"""Utilities for benchmark exploitation notebooks."""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.runner.utils import build_dataset, build_model, build_network
from genkit.metrics import metric_on_quantile, mssle, wasserstein_distance


MODEL_LABELS = {
    "gaussian_flow_linear": "GF-Linear",
    "alpha_stable_flow_linear": "ASF-Linear",
    "ddpm_v": "DDPM-V",
    "dlpm_eps_origin": "DLPM-Orig",
    "flow_matching_origin": "FM-Orig",
    "score_sde_origin": "ScoreSDE-Orig",
}
TRAIN_TAGS = {"short": "S.train", "long": "L.train"}
NETWORK_TAGS = {"mlp_small": "S.nn", "mlp_medium": "L.nn"}
METRIC_XIS = {"MSSLE_95": 0.95, "MSSLE_99": 0.99}
METRIC_ORDER = ["W1", "MSSLE_95", "MSSLE_99"]
GROUP_COLS = ["dataset_alpha", "model_name", "model_label", "model_alpha", "network_preset", "train_preset"]


def run_dirs(root: Path) -> list[Path]:
    """Return benchmark run directories sorted by numeric prefix."""
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name[:3].isdigit())


def pretty_model(name: str) -> str:
    """Map an internal model name to a short display label."""
    return MODEL_LABELS.get(name, name)


def setting_code(row: pd.Series) -> str:
    """Return the compact train/network/alpha code for one summary row."""
    pieces = [TRAIN_TAGS[row["train_preset"]], NETWORK_TAGS[row["network_preset"]]]
    if not pd.isna(row["model_alpha"]):
        pieces.append(f"a{float(row['model_alpha']):.3g}")
    return "(" + "/".join(pieces) + ")"


def to_latex_sci(x: float, digits: int = 2) -> str:
    """Format a scalar in compact LaTeX scientific notation."""
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10**exponent)
    if exponent == 0:
        return f"{x:.{digits}f}"
    return rf"{mantissa:.{digits}f}\,10^{{{exponent}}}"


def format_mean_std_latex(mean: float, std: float, caption: str = "", bold: bool = False) -> str:
    """Format a mean-plus-std pair as a LaTeX mathtext string."""
    mean_str = to_latex_sci(mean, digits=2)
    core = rf"\mathbf{{{mean_str}}}" if bold else mean_str
    if not np.isclose(std, 0.0):
        std_str = to_latex_sci(std, digits=2)
        core = rf"{core}_{{\pm {std_str}}}"
    if caption:
        core = rf"\underset{{\mathrm{{{caption}}}}}{{{core}}}"
    return rf"${core}$"


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


def load_generator(run_dir: Path, *, device: str = "cpu") -> tuple[Any, torch.Tensor, dict[str, Any]]:
    """Reload one trained generator and its test split from a run directory."""
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dtype_name = str(checkpoint["model_init"]["dtype"]).split(".")[-1]
    dtype = getattr(torch, dtype_name)
    network_cfg = yaml.safe_load(yaml.safe_dump(checkpoint["model_init"]["network"]))
    model_cfg = yaml.safe_load(yaml.safe_dump(checkpoint["model_init"]["model"]))
    model_cfg.setdefault("params", {})
    model_cfg["params"]["fdtype"] = dtype
    model_cfg["params"]["device"] = device

    x_train, _, x_test = build_dataset(config["dataset"], dtype=dtype, device=device)
    net, _ = build_network(network_cfg, x_train)
    net = net.to(device=device, dtype=dtype)
    generator, _ = build_model(model_cfg, net, x_train, dtype=dtype, device=device)
    generator._net.load_state_dict(checkpoint["network_state_dict"])
    generator._net.eval()
    return generator, x_test, config


def evaluate_run(run_dir: Path, *, n_samples: int, n_repeats: int, device: str = "cpu") -> list[dict[str, Any]]:
    """Evaluate one saved run several times and return one row per repeat."""
    generator, x_test, config = load_generator(run_dir, device=device)
    x_ref = x_test[: min(n_samples, len(x_test)), :]
    base_row = {
        "run_dir": run_dir.name,
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


def reduce_best_alpha(summary_exact: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Reduce alpha-parametrized models by best alpha at fixed data/train/network."""
    groups = ["dataset_alpha", "model_label", "network_preset", "train_preset"]
    rows = []
    for _, group in summary_exact.groupby(groups, dropna=False):
        best = group.iloc[0] if group["model_alpha"].isna().all() else group.nsmallest(1, f"{metric}_mean").iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def best_table_df(summary_exact: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pick the best setting for each dataset-alpha/model pair."""
    groups = ["dataset_alpha", "model_label"]
    rows = [group.nsmallest(1, f"{metric}_mean").iloc[0].to_dict() for _, group in summary_exact.groupby(groups, dropna=False)]
    return pd.DataFrame(rows)


def aggregated_best_alpha(summary_exact: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average each condition over dataset alpha after best-alpha reduction."""
    reduced = reduce_best_alpha(summary_exact, metric)
    rows = []
    for keys, group in reduced.groupby(["model_label", "network_preset", "train_preset"], dropna=False):
        row = dict(zip(["model_label", "network_preset", "train_preset"], keys, strict=True))
        row[metric] = float(group[f"{metric}_mean"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
