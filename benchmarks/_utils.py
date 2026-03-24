"""Utilities for benchmark run artifacts under _results/."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


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
