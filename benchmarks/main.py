#!/usr/bin/env python
"""Run benchmark sweeps from one YAML config."""

import argparse
import copy
from datetime import datetime
import itertools
import json
import logging
from pathlib import Path
import traceback
from typing import Any

import pandas as pd
import torch
import yaml

from genkit.datasets import fetch_real_data, fetch_synthetic_data, list_datasets
from genkit.diffusion import DDPMV, DDPMX0, DLPMEps
from genkit.flow import GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT
from genkit.nn import MLPModel, UNetModel
from genkit.thirdparty import DLPMEpsOrigin, FlowMatchingOrigin, ScoreSDEOrigin
from genkit.training import train
from genkit.visitor import CoreMetricsVisitor
from labkit.config import parse_dtype
from labkit.utils import set_seed


MODEL_REGISTRY = {
    "ddpm_v": DDPMV,
    "ddpm_x0": DDPMX0,
    "gaussian_flow_linear": GaussianFlowLinear,
    "gaussian_flow_ot": GaussianFlowOT,
    "gaussian_flow_ddpm": GaussianFlowDDPM,
    "dlpm_eps": DLPMEps,
    "dlpm_eps_origin": DLPMEpsOrigin,
    "flow_matching_origin": FlowMatchingOrigin,
    "score_sde_origin": ScoreSDEOrigin,
}


def setup_logging() -> None:
    """Configure a simple console logger."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML config file."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to a mapping.")
    return data


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one required config section."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def to_serializable(value: Any) -> Any:
    """Convert config values to JSON/YAML-safe objects."""
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return float(value.item()) if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    return value


def _slug(value: Any) -> str:
    """Convert one config value to a short path-safe slug."""
    return str(value).strip().replace("/", "-").replace(" ", "_").replace(".", "p")


def _grid_items(prefix: str, value: Any) -> list[tuple[str, Any]]:
    """Flatten one nested mapping into dotted leaf paths."""
    if not isinstance(value, dict):
        return [(prefix, value)]
    items: list[tuple[str, Any]] = []
    for key, child in value.items():
        items.extend(_grid_items(f"{prefix}.{key}" if prefix else str(key), child))
    return items


def _assign_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign one value inside a nested mapping using a dotted path."""
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _expand_entry_grid(entry_name: str, entry_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one named preset over all list-valued leaves."""
    static_cfg: dict[str, Any] = {}
    varying_items: list[tuple[str, list[Any]]] = []
    for key, value in _grid_items("", copy.deepcopy(entry_cfg)):
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Entry {entry_name!r} has an empty grid at {key!r}.")
            varying_items.append((key, value))
        else:
            _assign_path(static_cfg, key, value)

    if not varying_items:
        return [{"entry_name": entry_name, "config": static_cfg, "variant_name": entry_name, "variant_params": {}}]

    variants = []
    keys = [key for key, _ in varying_items]
    for combo in itertools.product(*(choices for _, choices in varying_items)):
        variant_cfg = copy.deepcopy(static_cfg)
        variant_params = dict(zip(keys, combo, strict=True))
        for key, value in variant_params.items():
            _assign_path(variant_cfg, key, value)
        suffix = "__".join(f"{key.split('.')[-1]}-{_slug(value)}" for key, value in variant_params.items())
        variants.append(
            {
                "entry_name": entry_name,
                "config": variant_cfg,
                "variant_name": f"{entry_name}__{suffix}",
                "variant_params": variant_params,
            }
        )
    return variants


def select_entries(section_name: str, registry: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """Select and expand named presets for one registry section."""
    variants = []
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown {section_name} entry {name!r}. Available: {sorted(registry)}")
        entry_cfg = registry[name]
        if not isinstance(entry_cfg, dict):
            raise ValueError(f"{section_name}.{name} must be a mapping.")
        variants.extend(_expand_entry_grid(name, entry_cfg))
    return variants


def make_batch_dir(run_cfg: dict[str, Any], save_cfg: dict[str, Any]) -> Path:
    """Create the parent output directory for one config sweep."""
    root = Path(save_cfg.get("root_dir", "runs")).expanduser()
    name = str(run_cfg.get("name", "run")).strip() or "run"
    batch_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def _make_run_dir(batch_dir: Path, combo_index: int, combo_name: str) -> Path:
    """Create one run subdirectory inside the batch directory."""
    run_dir = batch_dir / f"{combo_index:03d}_{combo_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_dataset(dataset_cfg: dict[str, Any], dtype: torch.dtype, device: str | torch.device):
    """Load one dataset split triplet from config."""
    kind = str(dataset_cfg.get("kind", "synthetic")).lower()
    name = str(dataset_cfg.get("name", "")).strip()
    if not name:
        raise ValueError("dataset.name must be provided.")
    kwargs = {**copy.deepcopy(dataset_cfg.get("params", {})), **copy.deepcopy(dataset_cfg.get("split", {})), "dtype": dtype, "device": device}
    if kind == "synthetic":
        return fetch_synthetic_data(name, **kwargs)
    if kind == "real":
        return fetch_real_data(name, **kwargs)
    raise ValueError(f"Unknown dataset.kind={kind!r}. Use 'synthetic' or 'real'.")


def build_network(network_cfg: dict[str, Any], x_train: torch.Tensor) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate one network backbone and return its resolved params."""
    name = str(network_cfg.get("name", "mlp")).lower()
    params = copy.deepcopy(network_cfg.get("params", {}))
    if name in {"mlp", "mlp_plain"}:
        params.setdefault("dim", int(x_train.shape[-1]))
        if name == "mlp_plain":
            params.setdefault("use_norm", False)
        return MLPModel(**params), params
    if name == "unet":
        if x_train.ndim != 4:
            raise ValueError(f"UNetModel expects 4D image-like data, got training shape {tuple(x_train.shape)}.")
        params.setdefault("in_channels", int(x_train.shape[1]))
        params.setdefault("out_channels", int(x_train.shape[1]))
        return UNetModel(**params), params
    raise ValueError(f"Unknown network.name={name!r}. Available: mlp, mlp_plain, unet.")


def build_model(
    model_cfg: dict[str, Any],
    net: torch.nn.Module,
    x_train: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate one generative model and return its resolved params."""
    name = str(model_cfg.get("name", "")).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model.name={name!r}. Available: {sorted(MODEL_REGISTRY)}.")
    params = copy.deepcopy(model_cfg.get("params", {}))
    params.setdefault("dim", int(x_train.shape[-1]))
    params.setdefault("fdtype", dtype)
    params.setdefault("device", device)
    return MODEL_REGISTRY[name](net=net, **params), params


def _epoch_frame(core_records: dict[str, Any]) -> pd.DataFrame:
    """Convert core visitor histories to a per-epoch DataFrame."""
    keys = ["training_loss", "training_loss_std", "grad_variance_epoch", "grad_norm_epoch"]
    max_len = max(len(core_records.get(key, [])) for key in keys)
    return pd.DataFrame(
        [
            {
                "epoch": idx + 1,
                **{key: core_records.get(key, [])[idx] if idx < len(core_records.get(key, [])) else float("nan") for key in keys},
            }
            for idx in range(max_len)
        ]
    )


def _resolved_config(
    config_path: Path,
    dtype: torch.dtype,
    batch_dir: Path,
    dataset_cfg: dict[str, Any],
    network_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    save_cfg: dict[str, Any],
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    net: torch.nn.Module,
) -> dict[str, Any]:
    """Build the saved resolved config for one run."""
    resolved = {
        "run": {
            "resolved_config_path": str(config_path.resolve()),
            "resolved_batch_dir": str(batch_dir.resolve()),
            "resolved_dtype": str(dtype),
        },
        "dataset": copy.deepcopy(dataset_cfg),
        "network": copy.deepcopy(network_cfg),
        "model": copy.deepcopy(model_cfg),
        "train": copy.deepcopy(train_cfg),
        "save": copy.deepcopy(save_cfg),
    }
    resolved["dataset"]["resolved_train_shape"] = list(x_train.shape)
    resolved["dataset"]["resolved_val_shape"] = list(x_val.shape)
    resolved["dataset"]["resolved_test_shape"] = list(x_test.shape)
    resolved["network"]["resolved_param_count"] = int(sum(param.numel() for param in net.parameters()))
    return to_serializable(resolved)


def _write_run_summary(
    run_dir: Path,
    resolved_config: dict[str, Any],
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    net: torch.nn.Module,
    diagnostics: dict[str, Any],
) -> None:
    """Write one short human-readable run summary."""
    core = diagnostics.get("visitors", {}).get("core", {})
    losses = core.get("training_loss", [])
    grad_norms = core.get("grad_norm_epoch", [])
    lines = [
        f"run_dir: {run_dir}",
        f"dataset: {resolved_config['dataset']['name']}",
        f"network: {resolved_config['network']['name']}",
        f"model: {resolved_config['model']['name']}",
        f"train_preset: {resolved_config['train'].get('preset_name', 'unknown')}",
        f"train_shape: {tuple(x_train.shape)}",
        f"val_shape: {tuple(x_val.shape)}",
        f"test_shape: {tuple(x_test.shape)}",
        f"n_parameters: {sum(param.numel() for param in net.parameters())}",
        f"final_loss: {losses[-1] if losses else 'nan'}",
        f"final_grad_norm: {grad_norms[-1] if grad_norms else 'nan'}",
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(
    config_path: Path,
    batch_dir: Path,
    combo_index: int,
    combo_name: str,
    run_cfg: dict[str, Any],
    dataset_variant: dict[str, Any],
    network_variant: dict[str, Any],
    model_variant: dict[str, Any],
    train_variant: dict[str, Any],
    trial_idx: int,
    save_cfg: dict[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Execute one config combination and save its artifacts."""
    run_dir = _make_run_dir(batch_dir, combo_index, combo_name)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = {**copy.deepcopy(dataset_variant["config"]), "preset_name": dataset_variant["variant_name"]}
    network_cfg = {**copy.deepcopy(network_variant["config"]), "preset_name": network_variant["variant_name"]}
    model_cfg = {**copy.deepcopy(model_variant["config"]), "preset_name": model_variant["variant_name"]}
    train_cfg = {**copy.deepcopy(train_variant["config"]), "preset_name": train_variant["variant_name"]}

    requested_config = to_serializable(
        {
            "run": {**copy.deepcopy(run_cfg), "trial_idx": int(trial_idx)},
            "dataset": dataset_cfg,
            "network": network_cfg,
            "model": model_cfg,
            "train": train_cfg,
            "save": copy.deepcopy(save_cfg),
        }
    )
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(requested_config, handle, sort_keys=False)

    set_seed(int(run_cfg.get("seed", 0)) + combo_index - 1)
    device = str(train_cfg.get("device", "cpu"))
    x_train, x_val, x_test = build_dataset(dataset_cfg, dtype=dtype, device=device)
    net, network_params = build_network(network_cfg, x_train)
    net = net.to(device=device, dtype=dtype)
    generative_model, model_params = build_model(model_cfg, net, x_train, dtype=dtype, device=device)

    train_kwargs = copy.deepcopy(train_cfg)
    train_kwargs.pop("preset_name", None)
    train_kwargs["device"] = device
    train_kwargs["ckpt_dir"] = str(ckpt_dir)

    logging.info(f"[{combo_index:03d}] run_dir: {run_dir}")
    logging.info(
        f"[{combo_index:03d}] dataset={dataset_variant['variant_name']} "
        f"network={network_variant['variant_name']} "
        f"model={model_variant['variant_name']} "
        f"train={train_variant['variant_name']}"
    )

    try:
        _, diagnostics = train(generative_model=generative_model, target_data=x_train, visitors=[CoreMetricsVisitor()], **train_kwargs)
        resolved_config = _resolved_config(
            config_path,
            dtype,
            batch_dir,
            dataset_cfg,
            network_cfg,
            model_cfg,
            train_cfg,
            save_cfg,
            x_train,
            x_val,
            x_test,
            net,
        )
        model_init = {
            "network": {"name": str(network_cfg.get("name")), "params": to_serializable(network_params)},
            "model": {"name": str(model_cfg.get("name")), "params": to_serializable(model_params)},
            "dtype": str(dtype),
            "device": device,
            "train_shape": list(x_train.shape),
        }
        checkpoint = {
            "model_name": str(model_cfg.get("name")),
            "network_name": str(network_cfg.get("name")),
            "model_init": model_init,
            "network_state_dict": net.state_dict(),
            "diagnostics": diagnostics,
        }
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_config, handle, sort_keys=False)
        (run_dir / "model_init.json").write_text(json.dumps(to_serializable(model_init), indent=2), encoding="utf-8")
        torch.save(checkpoint, run_dir / "checkpoint.pt")

        core_records = diagnostics.get("visitors", {}).get("core", {})
        _epoch_frame(core_records).to_csv(run_dir / "train_stats.csv", index=False)
        (run_dir / "train_stats.json").write_text(json.dumps(to_serializable(core_records), indent=2), encoding="utf-8")
        _write_run_summary(run_dir, resolved_config, x_train, x_val, x_test, net, diagnostics)

        losses = core_records.get("training_loss", [])
        grad_norms = core_records.get("grad_norm_epoch", [])
        return {
            "combo_index": combo_index,
            "combo_name": combo_name,
            "run_dir": str(run_dir),
            "trial_idx": trial_idx,
            "dataset_preset": dataset_variant["variant_name"],
            "network_preset": network_variant["variant_name"],
            "model_preset": model_variant["variant_name"],
            "train_preset": train_variant["variant_name"],
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "final_loss": losses[-1] if losses else float("nan"),
            "final_grad_norm": grad_norms[-1] if grad_norms else float("nan"),
        }
    except Exception as exc:
        error_text = traceback.format_exc()
        (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
        (run_dir / "summary.txt").write_text(
            "\n".join(
                [
                    f"run_dir: {run_dir}",
                    f"dataset: {dataset_variant['variant_name']}",
                    f"network: {network_variant['variant_name']}",
                    f"model: {model_variant['variant_name']}",
                    f"train: {train_variant['variant_name']}",
                    "status: failed",
                    f"error_type: {type(exc).__name__}",
                    f"error_message: {exc}",
                ]
            ) + "\n",
            encoding="utf-8",
        )
        logging.exception("[%03d] run failed: %s", combo_index, run_dir)
        return {
            "combo_index": combo_index,
            "combo_name": combo_name,
            "run_dir": str(run_dir),
            "trial_idx": trial_idx,
            "dataset_preset": dataset_variant["variant_name"],
            "network_preset": network_variant["variant_name"],
            "model_preset": model_variant["variant_name"],
            "train_preset": train_variant["variant_name"],
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "final_loss": float("nan"),
            "final_grad_norm": float("nan"),
        }


def main() -> None:
    """Run the configured benchmark sweep."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run benchmark sweeps from one YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to one YAML config file.")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Optional existing/shared batch directory.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split the combination grid into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to execute.")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")

    config = load_yaml(args.config)
    run_cfg = require_section(config, "run")
    sweep_cfg = require_section(config, "sweep")
    datasets_cfg = require_section(config, "datasets")
    networks_cfg = require_section(config, "networks")
    models_cfg = require_section(config, "models")
    trains_cfg = require_section(config, "trains")
    save_cfg = require_section(config, "save")

    dtype = parse_dtype(str(run_cfg.get("dtype", "float32")))
    dataset_variants = select_entries("datasets", datasets_cfg, list(sweep_cfg.get("datasets", [])))
    network_variants = select_entries("networks", networks_cfg, list(sweep_cfg.get("networks", [])))
    model_variants = select_entries("models", models_cfg, list(sweep_cfg.get("models", [])))
    train_variants = select_entries("trains", trains_cfg, list(sweep_cfg.get("trains", [])))
    n_trial = int(run_cfg.get("n_trial", 1))

    if not all([dataset_variants, network_variants, model_variants, train_variants]):
        raise ValueError("The sweep must select at least one dataset, network, model, and train entry.")
    if n_trial < 1:
        raise ValueError("run.n_trial must be >= 1.")

    batch_dir = make_batch_dir(run_cfg, save_cfg) if args.batch_dir is None else args.batch_dir.expanduser()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"batch_dir: {batch_dir}")
    print(f"shard: {args.shard_index + 1}/{args.shard_count}")
    print(f"available datasets: {', '.join(list_datasets())}")

    manifest_rows = []
    combinations = itertools.product(dataset_variants, network_variants, model_variants, train_variants, range(n_trial))
    for combo_index, (dataset_variant, network_variant, model_variant, train_variant, trial_idx) in enumerate(combinations, start=1):
        if (combo_index - 1) % args.shard_count != args.shard_index:
            continue
        combo_name = "__".join(
            [
                dataset_variant["variant_name"],
                network_variant["variant_name"],
                model_variant["variant_name"],
                train_variant["variant_name"],
                f"trial-{trial_idx + 1:02d}",
            ]
        )
        manifest_rows.append(
            run_one(
                config_path=args.config,
                batch_dir=batch_dir,
                combo_index=combo_index,
                combo_name=combo_name,
                run_cfg=run_cfg,
                dataset_variant=dataset_variant,
                network_variant=network_variant,
                model_variant=model_variant,
                train_variant=train_variant,
                trial_idx=trial_idx,
                save_cfg=save_cfg,
                dtype=dtype,
            )
        )

    manifest_name = "manifest.csv" if args.shard_count == 1 else f"manifest_shard_{args.shard_index:03d}.csv"
    summary_name = "summary.txt" if args.shard_count == 1 else f"summary_shard_{args.shard_index:03d}.txt"
    pd.DataFrame(manifest_rows).to_csv(batch_dir / manifest_name, index=False)
    n_failed = sum(row.get("status") == "failed" for row in manifest_rows)
    with (batch_dir / summary_name).open("w", encoding="utf-8") as handle:
        handle.write(f"n_runs: {len(manifest_rows)}\n")
        handle.write(f"n_failed: {n_failed}\n")
        handle.write(f"config: {args.config.resolve()}\n")
        handle.write(f"batch_dir: {batch_dir.resolve()}\n")
        handle.write(f"shard_index: {args.shard_index}\n")
        handle.write(f"shard_count: {args.shard_count}\n")
    print("done")


if __name__ == "__main__":
    main()
