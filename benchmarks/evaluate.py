"""Batch evaluation entrypoint for saved benchmark runs."""

import argparse
from pathlib import Path
import traceback
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.main import build_dataset, build_model, build_network, setup_logging
from genkit.inspect import estimate_init_error, estimate_training_loss_error, model_est_jacobian_spectral_curve
from genkit.metrics import fid, mmd_rbf, mssle, sliced_wasserstein, tail_coverage_error


EVAL_METRIC_NAMES = [
    "FID",
    "MMD_RBF",
    "SLICED_WASSERSTEIN",
    "TAIL_COVERAGE_ERROR",
    "MSSLE",
]
MODEL_LABELS = {
    "gaussian_flow_linear": "GF-Linear",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "dlpm_eps_origin": "DLPM-Orig",
    "flow_matching_origin": "FM-Orig",
    "score_sde_origin": "ScoreSDE-Orig",
}


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


def pretty_model(name: str) -> str:
    """Map an internal model name to a short display label."""
    return MODEL_LABELS.get(name, name)


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

    final_checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    checkpoint = final_checkpoint if checkpoint_epoch is None else torch.load(checkpoint_path, map_location=device)
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
    state_dict = checkpoint["network_state_dict"] if "network_state_dict" in checkpoint else checkpoint["model_state"]
    generator._net.load_state_dict(state_dict)
    generator._net.eval()
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
    final_epoch = int(config["train"]["n_epochs"])
    checkpoint_epochs = sorted(set(available_checkpoint_epochs(run_dir) + [final_epoch]))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scalar_rows: list[dict[str, Any]] = []
    jacobian_payload: dict[str, Any] | None = None
    warnings: list[str] = []

    for requested_epoch in checkpoint_epochs:
        generator, _, x_test, resolved_config, resolved_epoch = load_generator_and_data(
            run_dir,
            device=device,
            checkpoint_epoch=None if requested_epoch == final_epoch else requested_epoch,
        )
        checkpoint_epoch = int(resolved_epoch if resolved_epoch is not None else final_epoch)
        dataset_params = resolved_config.get("dataset", {}).get("params", {})
        model_params = resolved_config.get("model", {}).get("params", {})
        base_row = {
            "run_dir": run_dir.name,
            "checkpoint_epoch": checkpoint_epoch,
            "trial_idx": int(resolved_config.get("run", {}).get("trial_idx", 0)),
            "dataset_name": resolved_config["dataset"]["name"],
            "dataset_alpha": dataset_params.get("alpha", np.nan),
            "dataset_dim": dataset_params.get("dim", np.nan),
            "model_name": resolved_config["model"]["name"],
            "model_label": pretty_model(resolved_config["model"]["name"]),
            "model_alpha": model_params.get("alpha", np.nan),
            "network_preset": resolved_config["network"]["preset_name"],
            "train_preset": resolved_config["train"]["preset_name"],
        }

        train_stats = pd.read_csv(run_dir / "train_stats.csv")
        if "epoch" in train_stats.columns:
            train_stats = train_stats[train_stats["epoch"] <= checkpoint_epoch].copy()
        for column in ["training_loss", "training_loss_std", "grad_variance_epoch", "grad_norm_epoch"]:
            if column not in train_stats.columns:
                continue
            for _, row in train_stats[["epoch", column]].dropna().iterrows():
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

        x_ref = x_test[: min(int(n_eval_samples), len(x_test)), :]
        x_ref_cpu = x_ref.detach().cpu()
        for eval_repeat_idx in range(int(n_eval_repeats)):
            x_gen_cpu = generator.sample(n_samples=len(x_ref)).detach().cpu()
            metric_values = {
                "FID": float(fid(x_ref_cpu, x_gen_cpu)),
                "MMD_RBF": float(mmd_rbf(x_ref_cpu, x_gen_cpu)),
                "SLICED_WASSERSTEIN": float(sliced_wasserstein(x_ref_cpu, x_gen_cpu)),
                "TAIL_COVERAGE_ERROR": float(tail_coverage_error(x_ref_cpu, x_gen_cpu)),
                "MSSLE": float(mssle(x_ref_cpu, x_gen_cpu)),
            }
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

        x_ref_inspect = x_test[: min(int(inspect_samples), len(x_test)), :]
        try:
            init_error = float(
                estimate_init_error(
                    generator,
                    x_ref_inspect,
                    n_samples=min(4096, len(x_ref_inspect)),
                    k=10,
                )
            )
            training_loss_error = float(
                estimate_training_loss_error(
                    generator,
                    x_ref_inspect,
                    n_batches=16,
                    batch_size=min(256, len(x_ref_inspect)),
                    loss_type="native",
                )
            )
        except ValueError as exc:
            if "inspect" not in str(exc).lower():
                raise
            warnings.append(f"inspect_skipped: {exc}")
        else:
            scalar_rows.extend(
                [
                    {
                        **base_row,
                        "source": "inspect",
                        "metric_name": "init_error",
                        "value": init_error,
                        "eval_repeat_idx": np.nan,
                        "epoch": np.nan,
                    },
                    {
                        **base_row,
                        "source": "inspect",
                        "metric_name": "training_loss_error",
                        "value": training_loss_error,
                        "eval_repeat_idx": np.nan,
                        "epoch": np.nan,
                    },
                ]
            )

        if checkpoint_epoch != final_epoch:
            continue

        x_probe = x_ref_inspect[: min(int(probe_size), len(x_ref_inspect)), :]
        try:
            jac_curve, t_grid = model_est_jacobian_spectral_curve(
                generator,
                x_probe,
                n_power_iter=8,
                max_n_steps=10,
            )
        except ValueError as exc:
            if "inspect" not in str(exc).lower():
                raise
            warnings.append(f"jacobian_skipped: {exc}")
        else:
            jacobian_payload = {
                "curve": np.asarray(jac_curve, dtype=float),
                "t_grid": np.asarray(t_grid, dtype=float),
                "checkpoint_epoch": checkpoint_epoch,
            }

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
        "metric_names": EVAL_METRIC_NAMES,
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


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate saved benchmark runs.")
    parser.add_argument("--batch-dir", type=Path, required=True, help="Path to one saved benchmark batch directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-eval-samples", type=int, default=10000)
    parser.add_argument("--n-eval-repeats", type=int, default=10)
    parser.add_argument("--inspect-samples", type=int, default=2048)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")
    if args.n_eval_repeats < 1:
        raise ValueError("--n-eval-repeats must be >= 1.")

    batch_dir = args.batch_dir.expanduser()
    if not batch_dir.exists():
        raise FileNotFoundError(f"Missing batch directory: {batch_dir}")

    artifact_batch_dir = (
        batch_dir
        if batch_dir.name.endswith("_preprocessed")
        else batch_dir.with_name(f"{batch_dir.name}_preprocessed")
    )
    artifact_batch_dir.mkdir(parents=True, exist_ok=True)

    runs = [run_dir for run_dir in run_dirs(batch_dir) if (run_dir / "checkpoint.pt").exists()]
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
        handle.write(f"source_batch_dir: {batch_dir.resolve()}\n")
        handle.write(f"artifact_batch_dir: {artifact_batch_dir.resolve()}\n")
        handle.write(f"device: {args.device}\n")
        handle.write(f"shard_index: {args.shard_index}\n")
        handle.write(f"shard_count: {args.shard_count}\n")
