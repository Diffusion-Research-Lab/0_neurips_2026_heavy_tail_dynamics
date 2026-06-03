"""Configuration parsing helpers."""

import os
import copy
from pathlib import Path
from typing import Any, Dict, Union
import yaml
import torch
from .utils import set_seed, get_device
DEFAULTS: Dict[str, Any] = {
    "authors_root": {"mode": "home_subpath", "subpath": ["src", "DLPM"], "env_var": "DLPM_AUTHORS_ROOT"},
    "n_trials": 5,
    "dim": 10,
    "n_samples": 10_000,
    "n_steps": 100,
    "batch_size": 512,
    "n_epochs": 150,
    "lr": 1e-3,
    "seed": 0,
    "alpha": 1.8,
    "target_data_type": "alpha_stable",
    "l_alpha_data": {"type": "linspace", "start": 0.999, "end": 1.999, "steps": 5},
    "l_alpha_generator": {"type": "linspace", "start": 0.999, "end": 1.999, "steps": 5},
    "grad_clip_norm": None,
    "use_adamw": False,
    "num_workers": 0,
    "device": "auto",
    "dtype": "float64",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_dtype(name: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    key = str(name).lower()
    if key not in m:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return m[key]


def parse_idtype(name: str) -> torch.dtype:
    m = {
        "int32": torch.int32,
        "int": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
    }
    key = str(name).lower()
    if key not in m:
        raise ValueError(f"Unsupported integer dtype: {name!r}")
    return m[key]


def _resolve_authors_root(auth_cfg: Dict[str, Any]) -> Path:
    mode = auth_cfg.get("mode", "home_subpath")
    if mode == "explicit":
        return Path(auth_cfg["path"]).expanduser().resolve()
    if mode == "home_subpath":
        sub = auth_cfg.get("subpath", [])
        return Path(__file__).resolve().home().joinpath(*sub)
    raise ValueError(f"Unknown authors_root mode: {mode!r}")


def _materialize_specials(x: Any, *, dtype: torch.dtype) -> Any:
    # Generic: converts any {"type": "linspace", ...} or {"type": "tensor", ...} anywhere in the YAML.
    if isinstance(x, dict):
        t = x.get("type", None)
        if t == "linspace":
            return torch.linspace(float(x["start"]), float(x["end"]), int(x["steps"]), dtype=dtype)
        if t == "tensor":
            return torch.tensor(x["values"], dtype=dtype)
        return {k: _materialize_specials(v, dtype=dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_materialize_specials(v, dtype=dtype) for v in x]
    return x


class Config:
    """Experimental configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        for k, v in data.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def set_up(self) -> None:
        if hasattr(self, "AUTHORS_ROOT") and hasattr(self, "authors_root_env_var"):
            os.environ[str(self.authors_root_env_var)] = str(self.AUTHORS_ROOT)
        if hasattr(self, "seed"):
            set_seed(self.seed)
        if hasattr(self, "fdtype"):
            torch.set_default_dtype(self.fdtype)
        elif hasattr(self, "dtype"):
            torch.set_default_dtype(self.dtype)
        return self


def load_config(path: Union[str, Path], defaults: Dict[str, Any] = DEFAULTS) -> Config:
    """Load an experimental configuration."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(defaults, raw)

    dtype = parse_dtype(merged.get("dtype", "float64"))
    fdtype = parse_dtype(merged.get("fdtype", merged.get("dtype", "float64")))
    idtype = parse_idtype(merged.get("idtype", "int32"))
    merged = _materialize_specials(merged, dtype=dtype)

    # resolve runtime fields (kept generic; no per-key testing beyond these)
    auth_cfg = merged.get("authors_root", {}) or {}
    merged["AUTHORS_ROOT"] = _resolve_authors_root(auth_cfg)
    merged["authors_root_env_var"] = auth_cfg.get("env_var", "DLPM_AUTHORS_ROOT")
    merged["device"] = get_device(merged.get("device", "auto"))
    merged["dtype"] = dtype
    merged["fdtype"] = fdtype
    merged["idtype"] = idtype

    return Config(merged)
