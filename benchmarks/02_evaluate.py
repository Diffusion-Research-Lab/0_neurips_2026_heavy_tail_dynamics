"""Batch evaluation entrypoint for saved benchmark runs."""

import argparse
import copy
import importlib
from pathlib import Path
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np                                                                                               # noqa
import pandas as pd                                                                                              # noqa
import torch                                                                                                     # noqa
import yaml                                                                                                      # noqa
from genkit.inspect import estimate_init_error, estimate_training_loss_error, model_est_jacobian_spectral_curve  # noqa
from genkit.metrics import fid, mmd_rbf, mssle, sliced_wasserstein, tail_coverage_error                          # noqa


_main = importlib.import_module("benchmarks.01_main")
build_dataset = _main.build_dataset
build_model = _main.build_model
build_network = _main.build_network
setup_logging = _main.setup_logging


EVAL_METRIC_NAMES = [
    "FID",
    "MMD_RBF",
    "SLICED_WASSERSTEIN",
    "TAIL_COVERAGE_ERROR",
    "TCE(90%)",
    "TCE(95%)",
    "TCE(99%)",
    "TCE(99.9%)",
    "MSSLE",
]

PILOT_SELECTION_METRIC_TEMPLATES = {
    "train": "INNER_LOSS_TRAIN",
    "val": "INNER_LOSS_VAL",
    "test": "INNER_LOSS_TEST",
}

TAIL_COVERAGE_METRICS = {
    "TCE(90%)": 0.10,
    "TCE(95%)": 0.05,
    "TCE(99%)": 0.01,
    "TCE(99.9%)": 0.001,
}

MODEL_LABELS = {
    "gaussian_flow_linear": "GF-Linear",
    "gaussian_flow_ot": "GF-OT",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "tedm_origin": "TEDM-Orig",
}


def run_dirs(root: Path) -> list[Path]:
    """Return benchmark run directories sorted by numeric prefix."""
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name[:3].isdigit())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one evaluation batch."""
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark runs.")
    parser.add_argument("--batch-dir", type=Path, required=True, help="Path to one benchmark artifact batch directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-eval-samples", type=int, default=10000)
    parser.add_argument("--n-eval-repeats", type=int, default=10)
    parser.add_argument("--inspect-samples", type=int, default=2048)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--max-fid-dim", type=int, default=2048)
    parser.add_argument("--max-mmd-dim", type=int, default=2048)
    parser.add_argument("--max-mmd-samples", type=int, default=2048)
    parser.add_argument("--max-inspect-dim", type=int, default=1024)
    parser.add_argument("--inspect-image-data", action="store_true")
    parser.add_argument("--selection-only", action="store_true", help="Only evaluate the model's own loss on one split.")
    parser.add_argument("--selection-split", choices=sorted(PILOT_SELECTION_METRIC_TEMPLATES), default="val")
    parser.add_argument("--selection-repeats", type=int, default=8)
    parser.add_argument("--selection-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if any evaluated run fails.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate evaluation CLI arguments."""
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")
    if args.n_eval_repeats < 1:
        raise ValueError("--n-eval-repeats must be >= 1.")
    if args.n_eval_samples < 2:
        raise ValueError("--n-eval-samples must be >= 2.")
    if args.max_mmd_samples < 2:
        raise ValueError("--max-mmd-samples must be >= 2.")
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be >= 1.")
    if args.selection_repeats < 1:
        raise ValueError("--selection-repeats must be >= 1.")
    if args.selection_batch_size < 1:
        raise ValueError("--selection-batch-size must be >= 1.")


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


def pretty_model(name: str) -> str:
    """Map an internal model name to a short display label."""
    return MODEL_LABELS.get(name, name)


def checkpoint_dtype(config: dict[str, Any], checkpoint: dict[str, Any]) -> torch.dtype:
    """Resolve the dtype used to train a saved run."""
    dtype_spec = config.get("run", {}).get("dtype")
    if dtype_spec is None:
        dtype_spec = checkpoint.get("model_init", {}).get("dtype", torch.float64)
    return getattr(torch, str(dtype_spec).split(".")[-1], torch.float64)


def restore_generator(
    checkpoint: dict[str, Any],
    network_cfg: dict[str, Any],
    model_cfg_template: dict[str, Any],
    x_train: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: str,
):
    """Rebuild one generator and load a checkpoint into it."""
    net, _ = build_network(network_cfg, x_train)
    net = net.to(device=device, dtype=dtype)
    model_cfg = copy.deepcopy(model_cfg_template)
    generator, _ = build_model(model_cfg, net, x_train, dtype=dtype, device=device)
    state_dict = checkpoint["network_state_dict"] if "network_state_dict" in checkpoint else checkpoint["model_state"]
    generator._net.load_state_dict(state_dict)
    generator._net.eval()
    return generator


def sample_generator_in_batches(generator: Any, n_samples: int, batch_size: int) -> torch.Tensor:
    """Generate samples without requiring the full evaluation batch to fit on the GPU."""
    chunks = []
    remaining = int(n_samples)
    batch_size = int(batch_size)
    while remaining > 0:
        n_batch = min(batch_size, remaining)
        with torch.no_grad():
            chunks.append(generator.sample(n_samples=n_batch).detach().cpu())
        remaining -= n_batch
    return torch.cat(chunks, dim=0)


def estimate_split_inner_loss(
    generator: Any,
    x_split_cpu: torch.Tensor,
    *,
    device: str,
    n_repeats: int,
    batch_size: int,
) -> list[float]:
    """Estimate the model's own objective on one fixed split."""
    if len(x_split_cpu) == 0:
        return []

    losses: list[float] = []
    dtype = x_split_cpu.dtype
    pin = str(device).startswith("cuda")
    was_training = generator._net.training
    generator._net.eval()
    try:
        with torch.no_grad():
            for _ in range(int(n_repeats)):
                weighted_sum = 0.0
                weight_count = 0
                for start in range(0, len(x_split_cpu), int(batch_size)):
                    x = x_split_cpu[start: start + int(batch_size)].to(device=device, dtype=dtype, non_blocking=pin)
                    loss = generator.loss(x)
                    if loss.ndim != 0:
                        raise ValueError(f"generative_model.loss must return a scalar, got shape {tuple(loss.shape)}")
                    n_batch = int(x.shape[0])
                    weighted_sum += float(loss.detach().cpu()) * n_batch
                    weight_count += n_batch
                losses.append(weighted_sum / max(weight_count, 1))
    finally:
        generator._net.train(was_training)
    return losses


def compute_test_metrics(
    x_ref_cpu: torch.Tensor,
    x_gen_cpu: torch.Tensor,
    *,
    feature_dim: int,
    max_fid_dim: int,
    max_mmd_dim: int,
    max_mmd_samples: int,
) -> tuple[dict[str, float], list[str]]:
    """Compute sample-quality metrics, leaving failed metrics as NaN."""
    values = {name: float("nan") for name in EVAL_METRIC_NAMES}
    warnings: list[str] = []

    def compute_one(name: str, metric_fn) -> None:
        try:
            values[name] = float(metric_fn())
        except Exception as exc:
            warnings.append(f"{name}_failed: {type(exc).__name__}: {exc}")

    if feature_dim <= int(max_fid_dim):
        compute_one("FID", lambda: fid(x_ref_cpu, x_gen_cpu))
    else:
        warnings.append(f"fid_skipped: feature_dim={feature_dim} exceeds max_fid_dim={max_fid_dim}")

    if feature_dim <= int(max_mmd_dim):
        mmd_n = min(len(x_ref_cpu), int(max_mmd_samples))
        compute_one("MMD_RBF", lambda: mmd_rbf(x_ref_cpu[:mmd_n], x_gen_cpu[:mmd_n]))
    else:
        warnings.append(f"mmd_skipped: feature_dim={feature_dim} exceeds max_mmd_dim={max_mmd_dim}")

    compute_one("SLICED_WASSERSTEIN", lambda: sliced_wasserstein(x_ref_cpu, x_gen_cpu))
    compute_one("TAIL_COVERAGE_ERROR", lambda: tail_coverage_error(x_ref_cpu, x_gen_cpu))
    for metric_name, exceedance_prob in TAIL_COVERAGE_METRICS.items():
        probs = torch.tensor([float(exceedance_prob)], dtype=x_ref_cpu.dtype, device=x_ref_cpu.device)
        compute_one(
            metric_name,
            lambda probs=probs: tail_coverage_error(
                x_ref_cpu,
                x_gen_cpu,
                probs=probs,
            ),
        )
    compute_one("MSSLE", lambda: mssle(x_ref_cpu, x_gen_cpu))
    return values, warnings


def load_generator_and_data(
    run_dir: Path,
    *,
    device: str = "cpu",
    checkpoint_epoch: int | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any], int | None]:
    """Reload one trained generator and reconstruct its train/test splits."""
    if checkpoint_epoch is None:
        checkpoint_path = run_dir / "checkpoint.pt"
        resolved_epoch = None
    else:
        checkpoint_path = run_dir / "checkpoints" / f"ckpt_epoch_{int(checkpoint_epoch):04d}.pt"
        if not checkpoint_path.exists():
            available = available_checkpoint_epochs(run_dir)
            raise FileNotFoundError(
                f"Checkpoint epoch {checkpoint_epoch} not found in {run_dir}. "
                f"Available epochs: {available}"
            )
        resolved_epoch = int(checkpoint_epoch)

    final_checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    checkpoint = final_checkpoint if checkpoint_epoch is None else torch.load(checkpoint_path, map_location="cpu")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dtype = checkpoint_dtype(config, final_checkpoint)
    torch.set_default_dtype(dtype)

    network_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["network"]))
    model_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["model"]))
    model_cfg.setdefault("params", {})
    model_cfg["params"]["fdtype"] = dtype
    model_cfg["params"]["device"] = device

    x_train, _, x_test = build_dataset(config["dataset"], dtype=dtype, device="cpu")
    generator = restore_generator(checkpoint, network_cfg, model_cfg, x_train, dtype=dtype, device=device)
    return generator, x_train, x_test, config, resolved_epoch


def evaluate_one_run(
    run_dir: Path,
    *,
    artifact_batch_dir: Path,
    device: str,
    n_eval_samples: int,
    n_eval_repeats: int,
    inspect_samples: int,
    probe_size: int,
    sample_batch_size: int,
    max_fid_dim: int,
    max_mmd_dim: int,
    max_mmd_samples: int,
    max_inspect_dim: int,
    inspect_image_data: bool,
    selection_only: bool,
    selection_split: str,
    selection_repeats: int,
    selection_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate one saved benchmark run and write per-run artifacts."""
    artifact_dir = artifact_batch_dir / run_dir.name
    scalars_path = artifact_dir / "scalars.csv.gz"
    jacobian_path = artifact_dir / "jacobian_last_epoch.npz"
    summary_path = artifact_dir / "summary.yaml"

    if summary_path.exists() and not overwrite:
        summary = yaml.safe_load(summary_path.read_text()) or {}
        if summary.get("status") == "ok":
            return {
                "run_dir": run_dir.name,
                "artifact_dir": str(artifact_dir),
                "status": "skipped",
                "n_scalar_rows": int(summary.get("n_scalar_rows", 0)),
                "n_checkpoints": int(summary.get("n_checkpoints", 0)),
            }

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    final_checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    dtype = checkpoint_dtype(config, final_checkpoint)
    torch.set_default_dtype(dtype)

    network_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["network"]))
    model_cfg_template = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["model"]))
    model_cfg_template.setdefault("params", {})
    model_cfg_template["params"]["fdtype"] = dtype
    model_cfg_template["params"]["device"] = device

    x_train, x_val, x_test = build_dataset(config["dataset"], dtype=dtype, device="cpu")
    image_like = x_test.ndim > 2
    feature_dim = int(x_test[:1].reshape(1, -1).shape[1])
    final_epoch = int(config["train"]["n_epochs"])
    checkpoint_epochs = sorted(set(available_checkpoint_epochs(run_dir) + [final_epoch]))
    train_stats = pd.read_csv(run_dir / "train_stats.csv") if (run_dir / "train_stats.csv").exists() else pd.DataFrame()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scalar_rows: list[dict[str, Any]] = []
    jacobian_payload: dict[str, Any] | None = None
    warnings: list[str] = []

    def warn_once(warning: str) -> None:
        if warning not in warnings:
            warnings.append(warning)

    selection_tensors = {
        "train": x_train,
        "val": x_val,
        "test": x_test,
    }
    selection_metric_name = PILOT_SELECTION_METRIC_TEMPLATES[selection_split]

    for requested_epoch in checkpoint_epochs:
        if requested_epoch == final_epoch:
            checkpoint = final_checkpoint
            resolved_epoch = None
        else:
            checkpoint_path = run_dir / "checkpoints" / f"ckpt_epoch_{int(requested_epoch):04d}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            resolved_epoch = int(requested_epoch)
        generator = restore_generator(checkpoint, network_cfg, model_cfg_template, x_train, dtype=dtype, device=device)
        checkpoint_epoch = int(resolved_epoch if resolved_epoch is not None else final_epoch)
        dataset_params = config.get("dataset", {}).get("params", {})
        model_params = config.get("model", {}).get("params", {})
        base_row = {
            "run_dir": run_dir.name,
            "checkpoint_epoch": checkpoint_epoch,
            "trial_idx": int(config.get("run", {}).get("trial_idx", 0)),
            "dataset_preset": config["dataset"].get("preset_name", config["dataset"]["name"]),
            "dataset_name": config["dataset"]["name"],
            "dataset_alpha": dataset_params.get("alpha", np.nan),
            "dataset_dim": dataset_params.get("dim", np.nan),
            "model_name": config["model"]["name"],
            "model_label": pretty_model(config["model"]["name"]),
            "model_preset": config["model"].get("preset_name", config["model"]["name"]),
            "model_alpha": model_params.get("alpha", np.nan),
            "network_preset": config["network"]["preset_name"],
            "train_preset": config["train"]["preset_name"],
        }

        if selection_only and checkpoint_epoch != final_epoch:
            continue

        epoch_stats = train_stats
        if "epoch" in train_stats.columns:
            epoch_stats = train_stats[train_stats["epoch"] <= checkpoint_epoch]
        for column in ["training_loss", "training_loss_std", "grad_variance_epoch", "grad_norm_epoch"]:
            if column not in epoch_stats.columns:
                continue
            for _, row in epoch_stats[["epoch", column]].dropna().iterrows():
                scalar_rows.append(
                    {
                        **base_row,
                        "source": "train_stats",
                        "metric_name": column,
                        "value": float(row[column]),
                        "eval_repeat_idx": np.nan,
                        "epoch": int(row["epoch"]),
                    }
                )

        if selection_only:
            split_losses = estimate_split_inner_loss(
                generator,
                selection_tensors[selection_split].detach().cpu(),
                device=device,
                n_repeats=selection_repeats,
                batch_size=selection_batch_size,
            )
            for eval_repeat_idx, metric_value in enumerate(split_losses):
                scalar_rows.append(
                    {
                        **base_row,
                        "source": "pilot_selection",
                        "metric_name": selection_metric_name,
                        "value": float(metric_value),
                        "eval_repeat_idx": eval_repeat_idx,
                        "epoch": np.nan,
                    }
                )
            continue

        x_ref = x_test[: min(int(n_eval_samples), len(x_test))]
        x_ref_cpu = x_ref.detach().cpu()
        for eval_repeat_idx in range(int(n_eval_repeats)):
            x_gen_cpu = sample_generator_in_batches(generator, len(x_ref), sample_batch_size)
            metric_values, metric_warnings = compute_test_metrics(
                x_ref_cpu,
                x_gen_cpu,
                feature_dim=feature_dim,
                max_fid_dim=max_fid_dim,
                max_mmd_dim=max_mmd_dim,
                max_mmd_samples=max_mmd_samples,
            )
            for warning in metric_warnings:
                warn_once(warning)
            for metric_name in EVAL_METRIC_NAMES:
                scalar_rows.append(
                    {
                        **base_row,
                        "source": "test_metrics",
                        "metric_name": metric_name,
                        "value": metric_values[metric_name],
                        "eval_repeat_idx": eval_repeat_idx,
                        "epoch": np.nan,
                    }
                )

        x_ref_inspect = x_test[: min(int(inspect_samples), len(x_test))]
        can_inspect = len(x_ref_inspect) > 0 and int(inspect_samples) > 0 and (inspect_image_data or (not image_like and feature_dim <= int(max_inspect_dim)))
        if can_inspect:
            inspect_metrics = [
                (
                    "init_error",
                    lambda: estimate_init_error(
                        generator,
                        x_ref_inspect,
                        n_samples=min(4096, len(x_ref_inspect)),
                        k=10,
                    ),
                ),
                (
                    "training_loss_error",
                    lambda: estimate_training_loss_error(
                        generator,
                        x_ref_inspect,
                        n_batches=16,
                        batch_size=min(256, len(x_ref_inspect)),
                        loss_type="mse",
                    ),
                ),
            ]
            for metric_name, metric_fn in inspect_metrics:
                try:
                    metric_value = float(metric_fn())
                except Exception as exc:
                    reason = "skipped" if isinstance(exc, ValueError) and "inspect" in str(exc).lower() else "failed"
                    warn_once(f"inspect_{metric_name}_{reason}: {type(exc).__name__}: {exc}")
                    continue
                scalar_rows.append(
                    {
                        **base_row,
                        "source": "inspect",
                        "metric_name": metric_name,
                        "value": metric_value,
                        "eval_repeat_idx": np.nan,
                        "epoch": np.nan,
                    }
                )
        else:
            reason = "image_data" if image_like and not inspect_image_data else f"feature_dim={feature_dim}"
            warn_once(f"inspect_skipped: {reason}")

        if checkpoint_epoch != final_epoch:
            continue

        x_probe = x_ref_inspect[: min(int(probe_size), len(x_ref_inspect))]
        can_probe = len(x_probe) > 0 and int(probe_size) > 0 and (inspect_image_data or (not image_like and feature_dim <= int(max_inspect_dim)))
        if can_probe:
            try:
                jac_curve = model_est_jacobian_spectral_curve(
                    generator,
                    x_probe,
                    n_power_iter=8,
                    max_n_steps=10,
                )
            except Exception as exc:
                reason = "skipped" if isinstance(exc, ValueError) and "inspect" in str(exc).lower() else "failed"
                warn_once(f"jacobian_{reason}: {type(exc).__name__}: {exc}")
            else:
                jacobian_payload = {
                    "curve": np.asarray(jac_curve, dtype=float),
                    "checkpoint_epoch": checkpoint_epoch,
                }
        else:
            reason = "image_data" if image_like and not inspect_image_data else f"feature_dim={feature_dim}"
            warn_once(f"jacobian_skipped: {reason}")

    pd.DataFrame(scalar_rows).to_csv(scalars_path, index=False, compression="gzip")
    if jacobian_payload is not None:
        np.savez_compressed(jacobian_path, **jacobian_payload)

    summary = {
        "status": "ok",
        "run_dir": run_dir.name,
        "source_run_dir": str(run_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "device": device,
        "n_checkpoints": len(checkpoint_epochs),
        "checkpoint_epochs": checkpoint_epochs,
        "n_eval_repeats": int(n_eval_repeats),
        "n_eval_samples": int(n_eval_samples),
        "inspect_samples": int(inspect_samples),
        "probe_size": int(probe_size),
        "sample_batch_size": int(sample_batch_size),
        "feature_dim": feature_dim,
        "image_like": bool(image_like),
        "selection_only": bool(selection_only),
        "selection_split": selection_split,
        "selection_metric_name": selection_metric_name if selection_only else None,
        "selection_repeats": int(selection_repeats),
        "selection_batch_size": int(selection_batch_size),
        "max_fid_dim": int(max_fid_dim),
        "max_mmd_dim": int(max_mmd_dim),
        "max_mmd_samples": int(max_mmd_samples),
        "max_inspect_dim": int(max_inspect_dim),
        "inspect_image_data": bool(inspect_image_data),
        "metric_names": [selection_metric_name] if selection_only else EVAL_METRIC_NAMES,
        "n_scalar_rows": len(scalar_rows),
        "scalars_path": str(scalars_path),
        "jacobian_path": str(jacobian_path) if jacobian_payload is not None else None,
        "warnings": warnings,
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return {
        "run_dir": run_dir.name,
        "artifact_dir": str(artifact_dir),
        "status": "ok",
        "n_scalar_rows": len(scalar_rows),
        "n_checkpoints": len(checkpoint_epochs),
    }


def write_manifest_summary(
    artifact_batch_dir: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    source_batch_dir: Path,
) -> None:
    """Persist one evaluation manifest and shard summary."""
    manifest_name = "manifest_eval.csv" if args.shard_count == 1 else f"manifest_eval_shard_{args.shard_index:03d}.csv"
    summary_name = "summary_eval.txt" if args.shard_count == 1 else f"summary_eval_shard_{args.shard_index:03d}.txt"
    pd.DataFrame(manifest_rows).to_csv(artifact_batch_dir / manifest_name, index=False)
    n_failed = sum(row.get("status") == "failed" for row in manifest_rows)
    n_done = sum(row.get("status") == "ok" for row in manifest_rows)
    n_skipped = sum(row.get("status") == "skipped" for row in manifest_rows)
    with (artifact_batch_dir / summary_name).open("w", encoding="utf-8") as handle:
        handle.write(f"n_runs: {len(manifest_rows)}\n")
        handle.write(f"n_done: {n_done}\n")
        handle.write(f"n_skipped: {n_skipped}\n")
        handle.write(f"n_failed: {n_failed}\n")
        handle.write(f"source_batch_dir: {source_batch_dir.resolve()}\n")
        handle.write(f"artifact_batch_dir: {artifact_batch_dir.resolve()}\n")
        handle.write(f"device: {args.device}\n")
        handle.write(f"shard_index: {args.shard_index}\n")
        handle.write(f"shard_count: {args.shard_count}\n")
    if args.fail_on_error and n_failed:
        raise SystemExit(1)


def main() -> int:
    setup_logging()
    args = parse_args()
    validate_args(args)

    batch_dir = args.batch_dir.expanduser()
    if not batch_dir.exists():
        raise FileNotFoundError(f"Missing batch directory: {batch_dir}")

    artifact_batch_dir = (
        batch_dir
        if batch_dir.name.endswith("_evaluate")
        else batch_dir.with_name(f"{batch_dir.name}_evaluate")
    )
    artifact_batch_dir.mkdir(parents=True, exist_ok=True)

    runs = [run_dir for run_dir in run_dirs(batch_dir) if (run_dir / "checkpoint.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"No completed runs with checkpoint.pt found under {batch_dir}")
    manifest_rows = []
    for run_index, run_dir in enumerate(runs, start=1):
        if (run_index - 1) % args.shard_count != args.shard_index:
            continue
        try:
            row = evaluate_one_run(
                run_dir,
                artifact_batch_dir=artifact_batch_dir,
                device=args.device,
                n_eval_samples=args.n_eval_samples,
                n_eval_repeats=args.n_eval_repeats,
                inspect_samples=args.inspect_samples,
                probe_size=args.probe_size,
                sample_batch_size=args.sample_batch_size,
                max_fid_dim=args.max_fid_dim,
                max_mmd_dim=args.max_mmd_dim,
                max_mmd_samples=args.max_mmd_samples,
                max_inspect_dim=args.max_inspect_dim,
                inspect_image_data=args.inspect_image_data,
                selection_only=args.selection_only,
                selection_split=args.selection_split,
                selection_repeats=args.selection_repeats,
                selection_batch_size=args.selection_batch_size,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            row = {
                "run_dir": run_dir.name,
                "artifact_dir": str(artifact_batch_dir / run_dir.name),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        manifest_rows.append(row)
    write_manifest_summary(
        artifact_batch_dir,
        manifest_rows=manifest_rows,
        args=args,
        source_batch_dir=batch_dir,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
