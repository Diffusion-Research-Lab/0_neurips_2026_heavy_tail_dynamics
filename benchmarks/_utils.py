"""Utilities for benchmark run artifacts and shared benchmark plotting helpers."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml
import torch
from genkit import DLPMEpsOrigin as DLPM
from genkit import FlowMatchingOrigin as LinearFlow
from genkit.loss import barron_loss


def to_serializable(x: Any) -> Any:
    """Convert common benchmark objects to JSON/YAML-friendly values."""
    if isinstance(x, dict):
        return {str(k): to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return float(x.item())
        return x.detach().cpu().tolist()
    if isinstance(x, torch.device):
        return str(x)
    if isinstance(x, torch.dtype):
        return str(x)
    return x


def create_run_dir(out_root: Path, benchmark_name: str) -> Path:
    """Create a timestamped run directory and update the benchmark LATEST pointer."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = out_root / benchmark_name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    latest = out_root / benchmark_name / "LATEST"
    latest.write_text(run_dir.name, encoding="utf-8")
    return run_dir


def resolve_run_dir(out_root: Path, benchmark_name: str, run_dir: Path | None = None) -> Path:
    """Resolve an explicit run directory or the benchmark LATEST pointer."""
    if run_dir is not None:
        return run_dir
    latest = out_root / benchmark_name / "LATEST"
    if not latest.exists():
        raise FileNotFoundError(f"No LATEST file found for benchmark '{benchmark_name}' in {out_root}")
    name = latest.read_text(encoding="utf-8").strip()
    path = out_root / benchmark_name / name
    if not path.exists():
        raise FileNotFoundError(f"LATEST points to missing run directory: {path}")
    return path


def write_artifacts(run_dir: Path, config: Dict[str, Any], results: Dict[str, Any], run_summary: str) -> None:
    """Write benchmark config, results, and summary files into a run directory."""
    (run_dir / "config.yml").write_text(yaml.safe_dump(to_serializable(config), sort_keys=False), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(to_serializable(results), indent=2), encoding="utf-8")
    (run_dir / "run.txt").write_text(run_summary, encoding="utf-8")


class BarronLossMixin:
    """Apply the Barron loss to a flow model."""

    def __init__(self, *args, alpha: float = 1.0, **kwargs):
        """Store the Barron alpha parameter."""
        super().__init__(*args, **kwargs)
        self._alpha = float(alpha)

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor, t: int):
        """Compute unreduced Barron losses."""
        return barron_loss(pred, target, alpha=self._alpha, reduction="none")

    def _reduce(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Average a batch of loss values."""
        return loss_values.mean()


class PreprocessedAsinhTargetMixin:
    """Apply an asinh target transform during training and invert it at sampling time."""

    def __init__(self, *args, scale: float = 1.0, **kwargs):
        """Store the asinh scale parameter."""
        super().__init__(*args, **kwargs)
        self._scale = float(scale)
        if self._scale <= 0:
            raise ValueError(f"scale must be > 0, got {self._scale}")

    def _transform(self, x):
        """Transform targets with an asinh map."""
        return (x / self._scale).asinh()

    def _inverse_transform(self, y):
        """Invert the asinh target transform."""
        return self._scale * y.sinh()

    def _precompute_loss(self, x, z, t=None):
        """Run loss precomputation on transformed targets."""
        return super()._precompute_loss(x=self._transform(x), z=z, t=t)

    def sample(self, n_samples: int):
        """Sample in transformed space and map back to data space."""
        return self._inverse_transform(super().sample(n_samples))


class PreprocessedGaussianFlowLinear(PreprocessedAsinhTargetMixin, LinearFlow):
    """Gaussian flow with an asinh-preprocessed target space."""

    pass


def _fmt_alabel(value: float) -> str:
    """Format an alpha value without trailing decimal noise."""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _alpha_tag(value: float) -> str:
    """Convert an alpha value into a filename-safe tag."""
    return _fmt_alabel(value).replace("-", "neg").replace(".", "p")


def _name_tag(name: str) -> str:
    """Convert a display name into a class-name-safe tag."""
    return (
        name.replace("-", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("+", "plus")
    )


def build_model_specs(cfg):
    """Build benchmark-05 model specifications from a config object."""
    base_specs = {
        "Gaussian": (LinearFlow, {}),
        "Gaussian + asinh(x)": (PreprocessedGaussianFlowLinear, {"scale": float(cfg.asinh_scale)}),
        "AlphaStable": (DLPM, {"alpha": float(cfg.stable_alpha_model)}),
    }

    specs = []
    for base_name, (base_cls, base_kwargs) in base_specs.items():
        for barron_alpha in cfg.barron_alphas:
            name = f"{_name_tag(base_name)}BarronAlpha{_alpha_tag(barron_alpha)}"
            cls = type(name, (BarronLossMixin, base_cls), {})
            specs.append(
                {
                    "name": name,
                    "cls": cls,
                    "source_distri": base_name,
                    "loss": rf"$\alpha = {_fmt_alabel(barron_alpha)}$",
                    "loss_alpha": float(barron_alpha),
                    "extra_kwargs": {**base_kwargs, "alpha": float(barron_alpha)},
                }
            )
    return specs


def summarize_metric_values(values):
    """Return mean and population standard deviation for a list of scalars."""
    values = [float(v) for v in values]
    n_values = max(len(values), 1)
    mean = float(sum(values) / n_values)
    std = float((sum((value - mean) ** 2 for value in values) / n_values) ** 0.5)
    return {"values": values, "mean": mean, "std": std}


def loss_linestyle(alpha_value: float):
    """Return the plotting linestyle associated with a Barron alpha value."""
    predefined = {
        2.0: "solid",
        1.0: "dashed",
        0.0: "dashdot",
        -500.0: (0, (3, 1, 1, 1, 1, 1)),
        -10000.0: "dotted",
    }
    if alpha_value in predefined:
        return predefined[alpha_value]
    fallback = ["solid", "dashed", "dashdot", "dotted"]
    return fallback[int(abs(hash(alpha_value))) % len(fallback)]


def source_color(name: str) -> str:
    """Return the plotting color associated with a source distribution label."""
    if name == "Gaussian":
        return "tab:blue"
    if name.startswith("Gaussian +"):
        return "tab:green"
    if name == "AlphaStable":
        return "tab:red"
    return "tab:gray"


def average_trial_curve(trials, key: str) -> np.ndarray:
    """Average a named diagnostic curve over trials after trimming to a common length."""
    curves = []
    for trial in trials:
        values = trial.get("diagnostics", {}).get("visitors", {}).get("core", {}).get(key, [])
        if values:
            curves.append(np.asarray(values, dtype=float))
    if not curves:
        return np.asarray([], dtype=float)
    min_len = min(len(curve) for curve in curves)
    curves = [curve[:min_len] for curve in curves]
    return np.mean(np.stack(curves, axis=0), axis=0)


def to_latex_sci(x: float, digits: int = 2) -> str:
    """Format a scalar in compact LaTeX scientific notation."""
    if x == 0:
        return "0"
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / (10 ** exponent)
    if exponent == 0:
        return f"{x:.{digits}f}"
    return rf"{mantissa:.{digits}f}\,10^{{{exponent}}}"


def format_mean_std_latex(mean: float, std: float, bold: bool = False) -> str:
    """Format a mean-plus-std pair as a LaTeX mathtext string."""
    mean_str = to_latex_sci(mean, digits=2)
    if np.isclose(std, 0.0):
        return rf"$\mathbf{{{mean_str}}}$" if bold else rf"${mean_str}$"
    std_str = to_latex_sci(std, digits=2)
    if bold:
        return rf"$\mathbf{{{mean_str}}}_{{\pm {std_str}}}$"
    return rf"${mean_str}_{{\pm {std_str}}}$"


def save_figure(fig, fig_dir: Path, stem: str):
    """Save a figure as PDF and PNG and return the output paths."""
    pdf_path = fig_dir / f"{stem}.pdf"
    png_path = fig_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    return [pdf_path, png_path]
