#!/usr/bin/env python
"""Run benchmark sweeps from one YAML config."""

import argparse
import copy
import itertools
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd  # noqa
import torch  # noqa
import yaml  # noqa
from benchmarks.utils import load_yaml, make_batch_dir, require_section, select_entries  # noqa
from benchmarks._real_data_cache import (  # noqa
    load_preprocessed_real_dataset,
    require_preprocessed_real_data,
)
from datakit import fetch_real_data, list_datasets as list_real_datasets  # noqa
from genkit.datasets import fetch_synthetic_data, list_datasets as list_synthetic_datasets  # noqa
from genkit.diffusion import DDPMV, DLPMEps  # noqa
from genkit.flow_matching import GaussianFlowLinear, GaussianFlowOT  # noqa
from genkit.nn import MLPModel, UNetModel  # noqa
from genkit.thirdparty import TEDMOrigin  # noqa
from genkit.training import train  # noqa
from genkit.visitor import CoreMetricsVisitor  # noqa
from labkit.config import parse_dtype  # noqa
from labkit.utils import set_seed  # noqa


MODEL_REGISTRY = {
    "ddpm_v": DDPMV,
    "gaussian_flow_linear": GaussianFlowLinear,
    "gaussian_flow_ot": GaussianFlowOT,
    "dlpm_eps": DLPMEps,
    "tedm_origin": TEDMOrigin,
}


def setup_logging() -> None:
    """Configure a simple console logger."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one benchmark sweep execution."""
    parser = argparse.ArgumentParser(description="Run benchmark sweeps from one YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to one YAML config file.")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Optional existing/shared batch directory.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split the combination grid into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to execute.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose final checkpoint already exists.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if any run in this shard fails.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate shard-related CLI arguments."""
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")


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


def build_dataset(dataset_cfg: dict[str, Any], dtype: torch.dtype, device: str | torch.device):
    """Load one dataset split triplet from config."""
    kind = str(dataset_cfg.get("kind", "synthetic")).lower()
    name = str(dataset_cfg.get("name", "")).strip()
    if not name:
        raise ValueError("dataset.name must be provided.")
    kwargs = {
        **copy.deepcopy(dataset_cfg.get("params", {})),
        **copy.deepcopy(dataset_cfg.get("split", {})),
        "dtype": dtype,
        "device": device,
    }
    if kind == "synthetic":
        return fetch_synthetic_data(name, **kwargs)
    if kind == "real":
        if require_preprocessed_real_data():
            return load_preprocessed_real_dataset(dataset_cfg, dtype=dtype, device=device)
        try:
            return load_preprocessed_real_dataset(dataset_cfg, dtype=dtype, device=device)
        except FileNotFoundError:
            pass
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
    sample_shape = int(x_train.shape[-1]) if x_train.ndim == 2 else tuple(int(axis) for axis in x_train.shape[1:])
    params.setdefault("dim", sample_shape)
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
    run_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    network_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    save_cfg: dict[str, Any],
    trial_idx: int,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    net: torch.nn.Module,
) -> dict[str, Any]:
    """Build the saved resolved config for one run."""
    resolved = {
        "run": {**copy.deepcopy(run_cfg), "trial_idx": int(trial_idx)},
        "dataset": copy.deepcopy(dataset_cfg),
        "network": copy.deepcopy(network_cfg),
        "model": copy.deepcopy(model_cfg),
        "train": copy.deepcopy(train_cfg),
        "save": copy.deepcopy(save_cfg),
    }
    resolved["run"]["resolved_config_path"] = str(config_path.resolve())
    resolved["run"]["resolved_batch_dir"] = str(batch_dir.resolve())
    resolved["run"]["resolved_dtype"] = str(dtype)
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


def load_run_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one benchmark config and return its required sections."""
    config = load_yaml(config_path)
    return (
        require_section(config, "run"),
        require_section(config, "sweep"),
        require_section(config, "datasets"),
        require_section(config, "networks"),
        require_section(config, "models"),
        require_section(config, "trains"),
        require_section(config, "save"),
    )


def resolve_variants(
    sweep_cfg: dict[str, Any],
    datasets_cfg: dict[str, Any],
    networks_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    trains_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve selected sweep entries into expanded variants."""
    return (
        select_entries("datasets", datasets_cfg, list(sweep_cfg.get("datasets", []))),
        select_entries("networks", networks_cfg, list(sweep_cfg.get("networks", []))),
        select_entries("models", models_cfg, list(sweep_cfg.get("models", []))),
        select_entries("trains", trains_cfg, list(sweep_cfg.get("trains", []))),
    )


def write_manifest_summary(
    batch_dir: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Persist one shard manifest and summary."""
    manifest_name = "manifest.csv" if args.shard_count == 1 else f"manifest_shard_{args.shard_index:03d}.csv"
    summary_name = "summary.txt" if args.shard_count == 1 else f"summary_shard_{args.shard_index:03d}.txt"
    pd.DataFrame(manifest_rows).to_csv(batch_dir / manifest_name, index=False)
    n_failed = sum(row.get("status") == "failed" for row in manifest_rows)
    n_skipped = sum(row.get("status") == "skipped" for row in manifest_rows)
    with (batch_dir / summary_name).open("w", encoding="utf-8") as handle:
        handle.write(f"n_runs: {len(manifest_rows)}\n")
        handle.write(f"n_skipped: {n_skipped}\n")
        handle.write(f"n_failed: {n_failed}\n")
        handle.write(f"config: {args.config.resolve()}\n")
        handle.write(f"batch_dir: {batch_dir.resolve()}\n")
        handle.write(f"shard_index: {args.shard_index}\n")
        handle.write(f"shard_count: {args.shard_count}\n")
    if args.fail_on_error and n_failed:
        raise SystemExit(1)


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
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Execute one config combination and save its artifacts."""
    run_dir = batch_dir / f"{combo_index:03d}_{combo_name}"
    result = {
        "combo_index": combo_index,
        "combo_name": combo_name,
        "run_dir": str(run_dir),
        "trial_idx": trial_idx,
        "dataset_preset": dataset_variant["variant_name"],
        "network_preset": network_variant["variant_name"],
        "model_preset": model_variant["variant_name"],
        "train_preset": train_variant["variant_name"],
        "status": "failed",
        "error_type": "",
        "error_message": "",
        "final_loss": float("nan"),
        "final_grad_norm": float("nan"),
    }

    if skip_existing and (run_dir / "checkpoint.pt").exists():
        logging.info("[%03d] skipping existing run: %s", combo_index, run_dir)
        return {**result, "status": "skipped"}

    if run_dir.exists() and not skip_existing:
        return {
            **result,
            "error_type": "FileExistsError",
            "error_message": f"Run directory already exists: {run_dir}",
        }

    run_dir.mkdir(parents=True, exist_ok=skip_existing)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        (run_dir / "error.txt").unlink(missing_ok=True)
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

        logging.info(f"[{combo_index:03d}] run_dir: {run_dir}")
        logging.info(
            f"[{combo_index:03d}] dataset={dataset_variant['variant_name']} "
            f"network={network_variant['variant_name']} "
            f"model={model_variant['variant_name']} "
            f"train={train_variant['variant_name']}"
        )

        set_seed(int(run_cfg.get("seed", 0)) + combo_index - 1)
        device = str(train_cfg.get("device", "cpu"))
        x_train, x_val, x_test = build_dataset(dataset_cfg, dtype=dtype, device="cpu")
        net, network_params = build_network(network_cfg, x_train)
        net = net.to(device=device, dtype=dtype)
        generative_model, model_params = build_model(model_cfg, net, x_train, dtype=dtype, device=device)

        train_kwargs = copy.deepcopy(train_cfg)
        train_kwargs.pop("preset_name", None)
        train_kwargs["device"] = device
        train_kwargs["ckpt_dir"] = str(ckpt_dir)

        _, diagnostics = train(generative_model=generative_model, target_data=x_train, visitors=[CoreMetricsVisitor()], **train_kwargs)
        resolved_config = _resolved_config(
            config_path,
            dtype,
            batch_dir,
            run_cfg,
            dataset_cfg,
            network_cfg,
            model_cfg,
            train_cfg,
            save_cfg,
            trial_idx,
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
            **result,
            "status": "ok",
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
            **result,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


if __name__ == "__main__":

    setup_logging()
    args = parse_args()
    validate_args(args)

    run_cfg, sweep_cfg, datasets_cfg, networks_cfg, models_cfg, trains_cfg, save_cfg = load_run_config(args.config)

    dtype = parse_dtype(str(run_cfg.get("dtype", "float64")))
    torch.set_default_dtype(dtype)
    dataset_variants, network_variants, model_variants, train_variants = resolve_variants(
        sweep_cfg,
        datasets_cfg,
        networks_cfg,
        models_cfg,
        trains_cfg,
    )
    n_trial = int(run_cfg.get("n_trial", 1))

    if not all([dataset_variants, network_variants, model_variants, train_variants]):
        raise ValueError("The sweep must select at least one dataset, network, model, and train entry.")
    if n_trial < 1:
        raise ValueError("run.n_trial must be >= 1.")

    batch_dir = make_batch_dir(run_cfg, save_cfg) if args.batch_dir is None else args.batch_dir.expanduser()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"batch_dir: {batch_dir}")
    print(f"shard: {args.shard_index + 1}/{args.shard_count}")
    available_datasets = sorted(list_synthetic_datasets() + list_real_datasets())
    print(f"available datasets: {', '.join(available_datasets)}")

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
                skip_existing=args.skip_existing,
            )
        )
    write_manifest_summary(batch_dir, manifest_rows=manifest_rows, args=args)
    print("done")
