"""Batch evaluation entrypoint for saved benchmark runs."""

import argparse
import copy
import importlib
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np                                                                                       # noqa
import pandas as pd                                                                                      # noqa
import torch                                                                                             # noqa
from torch import nn                                                                                     # noqa
import yaml                                                                                              # noqa
from benchmarks._real_data_cache import load_preprocessed_real_dataset_metadata, real_dataset_cache_key  # noqa
from genkit.metrics import mmd_rbf, tail_coverage_error                                                  # noqa

_main = importlib.import_module("benchmarks.01_main")
build_dataset = _main.build_dataset
build_model = _main.build_model
build_network = _main.build_network


SYNTHETIC_EVAL_SAMPLES = 1_000_000
TCE_QUANTILE_MIN = 90.0
TCE_QUANTILE_MAX = 99.99
TCE_N_QUANTILES = 20
TCE_ANCHOR_NAMES = {
    90.0: "TCE(90)",
    99.0: "TCE(99)",
    99.9: "TCE(99,9)",
    99.99: "TCE(99,99)",
}
IMAGE_CLASS_RECOVERY_DATASETS = {"cifar100_lt", "imagenet_lt"}
IMAGE_CLASS_RECOVERY_METRIC_NAMES = [
    "CLASS_RECOVERY_INDEX",
    "CLASS_HIST_TV",
    "CLASS_PROBE_TEST_ACC",
    "N_TEST_CLASSES",
    "N_GEN_CLASSES",
]
IMAGE_CLASS_RECOVERY_MIN_TEST_COUNT = 1
IMAGE_CLASSIFIER_VERSION = 1
IMAGE_CLASSIFIER_EPOCHS = 8
IMAGE_CLASSIFIER_BATCH_SIZE = 256
IMAGE_CLASSIFIER_LR = 1.0e-3

PILOT_SELECTION_METRIC_NAME = "validation_loss"


def format_tce_metric_name(quantile: float) -> str:
    for anchor, name in TCE_ANCHOR_NAMES.items():
        if np.isclose(float(quantile), anchor):
            return name
    label = f"{float(quantile):.4f}".rstrip("0").rstrip(".")
    return f"TCE({label})"


def tce_metric_specs() -> list[tuple[str, float, float]]:
    probs = np.geomspace(
        1.0 - TCE_QUANTILE_MIN / 100.0,
        1.0 - TCE_QUANTILE_MAX / 100.0,
        TCE_N_QUANTILES,
    )
    for anchor_quantile in TCE_ANCHOR_NAMES:
        anchor_prob = 1.0 - anchor_quantile / 100.0
        index = int(np.argmin(np.abs(np.log(probs) - np.log(anchor_prob))))
        probs[index] = anchor_prob
    specs = []
    for prob in probs:
        quantile = 100.0 * (1.0 - float(prob))
        specs.append((format_tce_metric_name(quantile), float(prob), quantile))
    return specs


TCE_METRIC_SPECS = tce_metric_specs()
TAIL_COVERAGE_METRICS = {name: exceedance_prob for name, exceedance_prob, _ in TCE_METRIC_SPECS}
EVAL_METRIC_NAMES = ["MMD_RBF", *TAIL_COVERAGE_METRICS]
TEST_VS_TEST_SOURCE = "test_vs_test_metrics"

MODEL_LABELS = {
    "gaussian_flow_linear_euler": "GF-Linear Euler",
    "gaussian_flow_linear_heun": "GF-Linear Heun",
    "gaussian_flow_linear": "GF-Linear",
    "ddpm_v_ddpm": "DDPM-V DDPM",
    "ddpm_v_ddim": "DDPM-V DDIM",
    "dlpm_eps_a17": "DLPM alpha=1.7",
    "dlpm_eps_a19": "DLPM alpha=1.9",
    "ddpm_v": "DDPM-V",
    "dlpm_eps": "DLPM",
    "tedm_origin_nu21": "TEDM nu=2.1",
    "tedm_origin_nu30": "TEDM nu=3.0",
    "tedm_origin": "TEDM-Orig",
}
MODEL_ORDER = [
    "gaussian_flow_linear_euler",
    "gaussian_flow_linear_heun",
    "ddpm_v_ddpm",
    "ddpm_v_ddim",
    "dlpm_eps_a17",
    "dlpm_eps_a19",
    "tedm_origin_nu21",
    "tedm_origin_nu30",
    "dlpm_eps",
    "tedm_origin",
    "gaussian_flow_linear",
    "ddpm_v",
]


def canonical_model_name(model_preset: str, model_name: str) -> str:
    for candidate in MODEL_ORDER:
        if model_preset.startswith(candidate):
            return candidate
    return model_name


def checkpoint_dtype(config: dict[str, Any], checkpoint: dict[str, Any]) -> torch.dtype:
    """Resolve the dtype used to train a saved run."""
    dtype_spec = config.get("run", {}).get("dtype")
    if dtype_spec is None:
        dtype_spec = checkpoint.get("model_init", {}).get("dtype", torch.float64)
    return getattr(torch, str(dtype_spec).split(".")[-1], torch.float64)


def dataset_kind(dataset_cfg: dict[str, Any]) -> str:
    return str(dataset_cfg.get("kind", "synthetic")).lower()


def resolve_eval_sample_count(dataset_cfg: dict[str, Any], test_split_n: int, requested_n: int | None) -> int:
    if requested_n is not None:
        return min(int(requested_n), int(test_split_n)) if dataset_kind(dataset_cfg) == "real" else int(requested_n)
    if dataset_kind(dataset_cfg) == "synthetic":
        return int(SYNTHETIC_EVAL_SAMPLES)
    return int(test_split_n)


def sample_synthetic_reference(
    dataset_cfg: dict[str, Any],
    x_train: torch.Tensor,
    *,
    n_samples: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    from genkit.datasets import SYNTHETIC_DATASETS

    name = str(dataset_cfg.get("name", "")).strip()
    entry = SYNTHETIC_DATASETS[name]
    params = copy.deepcopy(dataset_cfg.get("params", {}))
    params.pop("n_samples", None)
    sampling_kwargs = {"n_samples": int(n_samples), "device": "cpu", "dtype": dtype}
    for param_name, builder in entry.sampler_kwargs_builders.items():
        sampling_kwargs[param_name] = builder(params)
    if params:
        unexpected = ", ".join(sorted(params))
        raise TypeError(f"Unexpected synthetic dataset params for evaluation: {unexpected}.")

    x_ref = entry.sampler(**sampling_kwargs).detach().cpu()
    if bool((dataset_cfg.get("split") or {}).get("standardize", False)):
        mean = x_train.detach().cpu().to(dtype=torch.float64).mean(dim=0, keepdim=True)
        std = x_train.detach().cpu().to(dtype=torch.float64).std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-12)
        x_ref = ((x_ref.to(dtype=torch.float64) - mean) / std).to(dtype=dtype)
    return x_ref.contiguous()


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


def compute_test_metrics(
    x_ref_cpu: torch.Tensor,
    x_gen_cpu: torch.Tensor,
    *,
    max_mmd_samples: int,
    mmd_device: str | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Compute sample-quality metrics, leaving failed metrics as NaN."""
    values = {name: float("nan") for name in EVAL_METRIC_NAMES}
    warnings: list[str] = []

    def compute_one(name: str, metric_fn) -> None:
        try:
            values[name] = float(metric_fn())
        except Exception as exc:
            warnings.append(f"{name}_failed: {type(exc).__name__}: {exc}")

    mmd_n = min(len(x_ref_cpu), int(max_mmd_samples))
    compute_one(
        "MMD_RBF",
        lambda: mmd_rbf(
            x_ref_cpu[:mmd_n].to(device=mmd_device) if mmd_device else x_ref_cpu[:mmd_n],
            x_gen_cpu[:mmd_n].to(device=mmd_device) if mmd_device else x_gen_cpu[:mmd_n],
        ),
    )

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
    return values, warnings


def compute_test_vs_test_metrics(
    x_ref_cpu: torch.Tensor,
    *,
    max_mmd_samples: int,
    mmd_device: str | None = None,
    seed: int = 0,
) -> tuple[dict[str, float], list[str]]:
    """Compute reference test-vs-test metrics from two disjoint test subsets."""
    values = {"MMD_RBF": float("nan")}
    warnings: list[str] = []
    n_each = min(int(max_mmd_samples), int(len(x_ref_cpu)) // 2)
    if n_each < 2:
        warnings.append("test_vs_test_MMD_RBF_failed: ValueError: need at least four test samples")
        return values, warnings

    try:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(int(len(x_ref_cpu)), generator=generator)
        x_left = x_ref_cpu[indices[:n_each]]
        x_right = x_ref_cpu[indices[n_each: 2 * n_each]]
        values["MMD_RBF"] = float(
            mmd_rbf(
                x_left.to(device=mmd_device) if mmd_device else x_left,
                x_right.to(device=mmd_device) if mmd_device else x_right,
            )
        )
    except Exception as exc:
        warnings.append(f"test_vs_test_MMD_RBF_failed: {type(exc).__name__}: {exc}")
    return values, warnings


def metadata_class_labels(metadata: dict[str, Any], split_name: str) -> torch.Tensor | None:
    """Return single-label class ids for one metadata split, when available."""
    split_meta = (metadata.get("splits") or {}).get(split_name) or {}
    records = split_meta.get("records") or []
    labels = []
    for record in records:
        if not isinstance(record, dict) or record.get("class_id") is None:
            return None
        labels.append(int(record["class_id"]))
    if not labels:
        return None
    return torch.tensor(labels, dtype=torch.long)


class ImageClassProbeNet(nn.Module):
    """Small CNN probe used only for labeled image diversity diagnostics."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def predict_image_class_probe(probe: dict[str, Any], x_cpu: torch.Tensor, *, device: str, batch_size: int | None = None) -> torch.Tensor:
    """Predict original class ids with the trained image classifier probe."""
    model = probe["model"].to(device=device)
    class_ids = probe["class_ids"].detach().cpu()
    batch_size = int(batch_size or IMAGE_CLASSIFIER_BATCH_SIZE)
    predictions = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(x_cpu), batch_size):
                x = x_cpu[start: start + batch_size].to(device=device, dtype=torch.float32)
                pred_idx = model(x).argmax(dim=1).detach().cpu()
                predictions.append(class_ids[pred_idx])
    finally:
        model.train(was_training)
    return torch.cat(predictions, dim=0) if predictions else torch.empty(0, dtype=torch.long)


def compute_image_class_recovery_metrics(probe: dict[str, Any], x_gen_cpu: torch.Tensor, *, device: str) -> dict[str, float]:
    """Compute class recovery diagnostics from generated image samples."""
    predicted_classes = predict_image_class_probe(probe, x_gen_cpu, device=device)
    generated_classes = set(int(value) for value in torch.unique(predicted_classes).tolist())
    eligible_class_tensor = probe["eligible_classes"].detach().cpu().long()
    eligible_classes = set(int(value) for value in eligible_class_tensor.tolist())
    recovered = len(generated_classes & eligible_classes)
    n_test_classes = int(probe["n_test_classes"])
    class_index = {int(class_id): index for index, class_id in enumerate(eligible_class_tensor.tolist())}
    encoded = torch.tensor([class_index.get(int(value), -1) for value in predicted_classes.detach().cpu().long().tolist()], dtype=torch.long)
    valid = encoded.ge(0)
    gen_counts = torch.bincount(encoded[valid], minlength=int(eligible_class_tensor.numel())).float()
    if int(len(predicted_classes)) <= 0:
        histogram_tv = float("nan")
    else:
        gen_probs = gen_counts.float() / float(len(predicted_classes))
        outside_mass = max(0.0, 1.0 - float(gen_probs.sum().item()))
        histogram_tv = float(0.5 * (torch.abs(probe["test_probs"].float() - gen_probs).sum().item() + outside_mass))
    return {
        "CLASS_RECOVERY_INDEX": float(recovered / n_test_classes) if n_test_classes else float("nan"),
        "CLASS_HIST_TV": histogram_tv,
        "CLASS_PROBE_TEST_ACC": float(probe["test_acc"]),
        "N_TEST_CLASSES": float(n_test_classes),
        "N_GEN_CLASSES": float(len(generated_classes)),
    }


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark runs.")
    parser.add_argument("--batch-dir", type=Path, required=True, help="Training batch directory produced by 01_main.py.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-eval-samples", type=int, default=None, help="Optional cap. Defaults to 1,000,000 for synthetic and full test split for real data.")
    parser.add_argument("--n-eval-repeats", type=int, default=10)
    parser.add_argument("--sample-batch-size", type=int, default=5000)
    parser.add_argument("--max-mmd-samples", type=int, default=2000)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--selection-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--selection-repeats", type=int, default=1)
    parser.add_argument("--selection-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")
    if args.n_eval_samples is not None and args.n_eval_samples < 1:
        raise ValueError("--n-eval-samples must be >= 1")
    if args.n_eval_repeats < 1:
        raise ValueError("--n-eval-repeats must be >= 1")
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be >= 1")
    if args.max_mmd_samples < 2:
        raise ValueError("--max-mmd-samples must be >= 2")
    if args.selection_repeats < 1:
        raise ValueError("--selection-repeats must be >= 1")
    if args.selection_batch_size < 1:
        raise ValueError("--selection-batch-size must be >= 1")
    batch_dir = args.batch_dir.expanduser()
    if not batch_dir.exists():
        raise FileNotFoundError(f"Missing batch directory: {batch_dir}")

    artifact_batch_dir = (
        batch_dir
        if batch_dir.name.endswith("_evaluate")
        else batch_dir.with_name(f"{batch_dir.name}_evaluate")
    )
    artifact_batch_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        run_dir
        for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir() and path.name[:3].isdigit())
        if (run_dir / "checkpoint.pt").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"No completed runs with checkpoint.pt found under {batch_dir}")
    if args.selection_only:
        image_class_probe, image_class_probe_warnings = None, []
    else:
        reference_run_dir = runs[0]
        config = yaml.safe_load((reference_run_dir / "config.yaml").read_text())
        dataset_cfg = config["dataset"]
        dataset_name = str(dataset_cfg.get("name", ""))
        if dataset_name not in IMAGE_CLASS_RECOVERY_DATASETS:
            image_class_probe, image_class_probe_warnings = None, []
        else:
            try:
                final_checkpoint = torch.load(reference_run_dir / "checkpoint.pt", map_location="cpu")
                dtype = checkpoint_dtype(config, final_checkpoint)
                cache_key = real_dataset_cache_key(dataset_cfg, dtype)
                cache_path = reference_run_dir.parent.parent / "class_probes" / f"{dataset_name}_{cache_key}_v{IMAGE_CLASSIFIER_VERSION}.pt"
                if cache_path.is_file():
                    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if not isinstance(payload, dict) or int(payload.get("version", -1)) != IMAGE_CLASSIFIER_VERSION:
                        raise ValueError(f"classifier probe cache version mismatch: {cache_path}")
                    model = ImageClassProbeNet(int(payload["in_channels"]), int(payload["num_classes"]))
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    image_class_probe = {
                        "model": model,
                        "class_ids": payload["class_ids"].long(),
                        "eligible_classes": payload["eligible_classes"].long(),
                        "test_probs": payload["test_probs"].float(),
                        "test_acc": float(payload["test_acc"]),
                        "n_test_classes": int(payload["n_test_classes"]),
                    }
                    image_class_probe_warnings = []
                else:
                    x_train, x_test = build_dataset(dataset_cfg, dtype=dtype, device="cpu", splits=("train", "test"))
                    metadata = load_preprocessed_real_dataset_metadata(dataset_cfg, dtype=dtype)
                    train_labels = metadata_class_labels(metadata, "train")
                    test_labels = metadata_class_labels(metadata, "test")
                    if train_labels is None or test_labels is None:
                        raise ValueError("single-label class metadata is missing")

                    x_train_cpu = x_train.detach().cpu()
                    x_test_cpu = x_test.detach().cpu()
                    if x_train_cpu.ndim != 4:
                        raise ValueError(f"classifier probe expects image tensors with shape (N,C,H,W), got {tuple(x_train_cpu.shape)}")
                    if len(train_labels) != len(x_train_cpu):
                        raise ValueError(f"train label count {len(train_labels)} does not match x_train count {len(x_train_cpu)}")
                    if len(test_labels) != len(x_test_cpu):
                        raise ValueError(f"test label count {len(test_labels)} does not match x_test count {len(x_test_cpu)}")

                    cuda_devices: list[int] = []
                    if str(args.device).startswith("cuda") and torch.cuda.is_available():
                        device_obj = torch.device(args.device)
                        cuda_devices = [torch.cuda.current_device() if device_obj.index is None else int(device_obj.index)]

                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(0)
                        if cuda_devices:
                            torch.cuda.manual_seed_all(0)

                        train_labels = train_labels.detach().cpu().long()
                        test_labels = test_labels.detach().cpu().long()
                        class_ids = torch.unique(train_labels, sorted=True)
                        if class_ids.numel() < 2:
                            raise ValueError("classifier probe needs at least two train classes")

                        class_index = {int(class_id): index for index, class_id in enumerate(class_ids.tolist())}
                        y_train = torch.tensor([class_index.get(int(value), -1) for value in train_labels.tolist()], dtype=torch.long)
                        valid_train = y_train.ge(0)
                        x_train_cpu = x_train_cpu[valid_train]
                        y_train = y_train[valid_train]
                        if len(y_train) == 0:
                            raise ValueError("classifier probe has no labeled train samples")

                        model = ImageClassProbeNet(int(x_train_cpu.shape[1]), int(class_ids.numel())).to(device=args.device)
                        counts = torch.bincount(y_train, minlength=int(class_ids.numel())).float()
                        class_weights = (counts.sum() / counts.clamp_min(1.0)).sqrt()
                        class_weights = class_weights / class_weights.mean().clamp_min(1.0e-12)
                        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device=args.device))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=IMAGE_CLASSIFIER_LR, weight_decay=1.0e-4)
                        batch_size = min(int(IMAGE_CLASSIFIER_BATCH_SIZE), len(x_train_cpu))

                        model.train()
                        for _ in range(int(IMAGE_CLASSIFIER_EPOCHS)):
                            perm = torch.randperm(len(x_train_cpu))
                            for start in range(0, len(x_train_cpu), batch_size):
                                idx = perm[start: start + batch_size]
                                x = x_train_cpu[idx].to(device=args.device, dtype=torch.float32)
                                y = y_train[idx].to(device=args.device)
                                optimizer.zero_grad(set_to_none=True)
                                loss = loss_fn(model(x), y)
                                loss.backward()
                                optimizer.step()

                        test_pred = predict_image_class_probe({"model": model, "class_ids": class_ids}, x_test_cpu, device=args.device)
                        known_test = torch.tensor([class_index.get(int(value), -1) for value in test_labels.tolist()], dtype=torch.long).ge(0)
                        if known_test.any():
                            test_acc = float(test_pred[known_test].eq(test_labels[known_test]).float().mean().item())
                            known_test_labels = test_labels[known_test]
                        else:
                            test_acc = float("nan")
                            known_test_labels = test_labels[:0]

                        test_classes, test_counts = torch.unique(known_test_labels, sorted=True, return_counts=True)
                        eligible_classes = test_classes[test_counts >= int(IMAGE_CLASS_RECOVERY_MIN_TEST_COUNT)]
                        if eligible_classes.numel() == 0:
                            raise ValueError("classifier probe has no eligible test classes")

                        eligible_index = {int(class_id): index for index, class_id in enumerate(eligible_classes.tolist())}
                        encoded_test = torch.tensor([eligible_index.get(int(value), -1) for value in known_test_labels.tolist()], dtype=torch.long)
                        valid_test = encoded_test.ge(0)
                        test_hist = torch.bincount(encoded_test[valid_test], minlength=int(eligible_classes.numel())).float()
                        image_class_probe = {
                            "model": model.cpu(),
                            "class_ids": class_ids.cpu(),
                            "eligible_classes": eligible_classes.cpu(),
                            "test_probs": (test_hist / test_hist.sum().clamp_min(1.0)).cpu(),
                            "test_acc": test_acc,
                            "n_test_classes": int(eligible_classes.numel()),
                        }

                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "version": IMAGE_CLASSIFIER_VERSION,
                        "class_ids": image_class_probe["class_ids"].cpu(),
                        "eligible_classes": image_class_probe["eligible_classes"].cpu(),
                        "test_probs": image_class_probe["test_probs"].cpu(),
                        "test_acc": float(image_class_probe["test_acc"]),
                        "n_test_classes": int(image_class_probe["n_test_classes"]),
                        "state_dict": image_class_probe["model"].cpu().state_dict(),
                        "in_channels": int(image_class_probe["model"].net[0].in_channels),
                        "num_classes": int(image_class_probe["class_ids"].numel()),
                    }
                    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
                    torch.save(payload, tmp_path)
                    os.replace(tmp_path, cache_path)
                    image_class_probe_warnings = []
            except Exception as exc:
                image_class_probe = None
                image_class_probe_warnings = [f"class_recovery_probe_skipped: {type(exc).__name__}: {exc}"]
    manifest_rows = []
    for run_index, run_dir in enumerate(runs, start=1):
        if (run_index - 1) % args.shard_count != args.shard_index:
            continue
        try:
            artifact_dir = artifact_batch_dir / run_dir.name
            scalars_path = artifact_dir / "scalars.csv.gz"
            summary_path = artifact_dir / "summary.yaml"

            if summary_path.exists() and not args.overwrite:
                summary = yaml.safe_load(summary_path.read_text()) or {}
                if summary.get("status") == "ok":
                    can_skip = True
                    if args.selection_only:
                        can_skip = False
                        if scalars_path.exists():
                            sample = pd.read_csv(scalars_path, usecols=["source", "metric_name"])
                            can_skip = bool((sample["source"].eq("pilot_selection") & sample["metric_name"].eq(PILOT_SELECTION_METRIC_NAME)).any())
                    if can_skip:
                        row = {
                            "run_dir": run_dir.name,
                            "artifact_dir": str(artifact_dir),
                            "status": "skipped",
                            "n_scalar_rows": int(summary.get("n_scalar_rows", 0)),
                            "n_checkpoints": int(summary.get("n_checkpoints", 0)),
                        }
                        manifest_rows.append(row)
                        continue

            config = yaml.safe_load((run_dir / "config.yaml").read_text())
            train_stats = pd.read_csv(run_dir / "train_stats.csv") if (run_dir / "train_stats.csv").exists() else pd.DataFrame()
            if args.selection_only:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                final_checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
                dtype = checkpoint_dtype(config, final_checkpoint)
                torch.set_default_dtype(dtype)
                x_train, x_selection = build_dataset(
                    config["dataset"],
                    dtype=dtype,
                    device="cpu",
                    splits=("train", args.selection_split),
                )
                if len(x_selection) == 0:
                    raise ValueError(f"Selection split {args.selection_split!r} is empty.")

                network_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["network"]))
                model_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["model"]))
                model_cfg.setdefault("params", {})
                model_cfg["params"]["fdtype"] = dtype
                model_cfg["params"]["device"] = args.device
                net, _ = build_network(network_cfg, x_train)
                net = net.to(device=args.device, dtype=dtype)
                generator, _ = build_model(model_cfg, net, x_train, dtype=dtype, device=args.device)
                state_dict = final_checkpoint["network_state_dict"] if "network_state_dict" in final_checkpoint else final_checkpoint["model_state"]
                generator._net.load_state_dict(state_dict)
                generator._net.eval()

                final_epoch = int(config["train"]["n_epochs"])
                dataset_params = config.get("dataset", {}).get("params", {})
                model_params = config.get("model", {}).get("params", {})
                model_preset = config["model"].get("preset_name", config["model"]["name"])
                model_name = canonical_model_name(str(model_preset), str(config["model"]["name"]))
                feature_dim = int(x_selection[:1].reshape(1, -1).shape[1])
                scalar_rows: list[dict[str, Any]] = []
                base_row = {
                    "run_dir": run_dir.name,
                    "checkpoint_epoch": final_epoch,
                    "trial_idx": int(config.get("run", {}).get("trial_idx", 0)),
                    "dataset_preset": config["dataset"].get("preset_name", config["dataset"]["name"]),
                    "dataset_name": config["dataset"]["name"],
                    "dataset_alpha": dataset_params.get("alpha", np.nan),
                    "dataset_dim": dataset_params.get("dim", np.nan),
                    "model_name": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "model_preset": model_preset,
                    "model_alpha": model_params.get("alpha", np.nan),
                    "network_preset": config["network"]["preset_name"],
                    "train_preset": config["train"]["preset_name"],
                    "train_lr": config["train"].get("lr", np.nan),
                }
                stats = train_stats.copy()
                if not stats.empty and "epoch" in stats.columns:
                    stats["epoch"] = pd.to_numeric(stats["epoch"], errors="coerce")
                for column in ["training_loss", "grad_norm"]:
                    if stats.empty or "epoch" not in stats.columns or column not in stats.columns:
                        continue
                    for _, stat_row in stats[["epoch", column]].dropna().iterrows():
                        scalar_rows.append(
                            {
                                **base_row,
                                "source": "train_stats",
                                "metric_name": column,
                                "value": float(stat_row[column]),
                                "eval_repeat_idx": np.nan,
                                "epoch": int(stat_row["epoch"]),
                            }
                        )

                selection_values = []
                split_dtype = x_selection.dtype
                pin = str(args.device).startswith("cuda")
                with torch.no_grad():
                    for eval_repeat_idx in range(int(args.selection_repeats)):
                        weighted_sum = 0.0
                        weight_count = 0
                        for start in range(0, len(x_selection), int(args.selection_batch_size)):
                            x = x_selection[start: start + int(args.selection_batch_size)].to(
                                device=args.device,
                                dtype=split_dtype,
                                non_blocking=pin,
                            )
                            loss = generator.loss(x)
                            if loss.ndim != 0:
                                raise ValueError(f"generative_model.loss must return a scalar, got shape {tuple(loss.shape)}")
                            n_batch = int(x.shape[0])
                            weighted_sum += float(loss.detach().cpu()) * n_batch
                            weight_count += n_batch
                        value = weighted_sum / max(weight_count, 1)
                        selection_values.append(value)
                        scalar_rows.append(
                            {
                                **base_row,
                                "source": "pilot_selection",
                                "metric_name": PILOT_SELECTION_METRIC_NAME,
                                "value": float(value),
                                "eval_repeat_idx": eval_repeat_idx,
                                "epoch": np.nan,
                            }
                        )

                pd.DataFrame(scalar_rows).to_csv(scalars_path, index=False, compression="gzip")
                summary = {
                    "status": "ok",
                    "run_dir": run_dir.name,
                    "source_run_dir": str(run_dir.resolve()),
                    "artifact_dir": str(artifact_dir.resolve()),
                    "device": args.device,
                    "n_checkpoints": 1,
                    "checkpoint_epochs": [final_epoch],
                    "feature_dim": feature_dim,
                    "image_like": bool(x_selection.ndim > 2),
                    "selection_only": True,
                    "selection_split": args.selection_split,
                    "selection_metric_name": PILOT_SELECTION_METRIC_NAME,
                    "selection_repeats": int(args.selection_repeats),
                    "selection_batch_size": int(args.selection_batch_size),
                    "selection_score": float(np.mean(selection_values)),
                    "metric_names": [PILOT_SELECTION_METRIC_NAME],
                    "n_scalar_rows": len(scalar_rows),
                    "scalars_path": str(scalars_path),
                    "warnings": [],
                }
                summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
                manifest_rows.append(
                    {
                        "run_dir": run_dir.name,
                        "artifact_dir": str(artifact_dir),
                        "status": "ok",
                        "n_scalar_rows": len(scalar_rows),
                        "n_checkpoints": 1,
                    }
                )
                continue

            final_checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
            dtype = checkpoint_dtype(config, final_checkpoint)
            torch.set_default_dtype(dtype)

            network_cfg = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["network"]))
            model_cfg_template = yaml.safe_load(yaml.safe_dump(final_checkpoint["model_init"]["model"]))
            model_cfg_template.setdefault("params", {})
            model_cfg_template["params"]["fdtype"] = dtype
            model_cfg_template["params"]["device"] = args.device

            needed_splits = ("train", "test")
            loaded_splits = build_dataset(config["dataset"], dtype=dtype, device="cpu", splits=needed_splits)
            split_tensors = dict(zip(needed_splits, loaded_splits))
            x_train = split_tensors["train"]
            metric_ref = split_tensors["test"]
            image_like = metric_ref.ndim > 2
            feature_dim = int(metric_ref[:1].reshape(1, -1).shape[1])
            eval_sample_count = resolve_eval_sample_count(config["dataset"], len(metric_ref), args.n_eval_samples)
            if dataset_kind(config["dataset"]) == "synthetic":
                x_ref_cpu = sample_synthetic_reference(
                    config["dataset"],
                    x_train,
                    n_samples=eval_sample_count,
                    dtype=dtype,
                )
                eval_reference_source = "synthetic_resample"
            else:
                x_ref_cpu = metric_ref[:eval_sample_count].detach().cpu()
                eval_reference_source = "test_split"
            final_epoch = int(config["train"]["n_epochs"])
            checkpoint_epochs = []
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.exists():
                for path in ckpt_dir.glob("ckpt_epoch_*.pt"):
                    try:
                        checkpoint_epochs.append(int(path.stem.split("_")[-1]))
                    except ValueError:
                        continue
            checkpoint_epochs = sorted(set(checkpoint_epochs + [final_epoch]))
            artifact_dir.mkdir(parents=True, exist_ok=True)

            scalar_rows: list[dict[str, Any]] = []
            warnings: list[str] = list(image_class_probe_warnings)

            for requested_epoch in checkpoint_epochs:
                if requested_epoch == final_epoch:
                    checkpoint = final_checkpoint
                    resolved_epoch = None
                else:
                    checkpoint_path = run_dir / "checkpoints" / f"ckpt_epoch_{int(requested_epoch):04d}.pt"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    resolved_epoch = int(requested_epoch)

                net, _ = build_network(network_cfg, x_train)
                net = net.to(device=args.device, dtype=dtype)
                model_cfg = copy.deepcopy(model_cfg_template)
                generator, _ = build_model(model_cfg, net, x_train, dtype=dtype, device=args.device)
                state_dict = checkpoint["network_state_dict"] if "network_state_dict" in checkpoint else checkpoint["model_state"]
                generator._net.load_state_dict(state_dict)
                generator._net.eval()

                checkpoint_epoch = int(resolved_epoch if resolved_epoch is not None else final_epoch)
                dataset_params = config.get("dataset", {}).get("params", {})
                model_params = config.get("model", {}).get("params", {})
                model_preset = config["model"].get("preset_name", config["model"]["name"])
                model_name = canonical_model_name(str(model_preset), str(config["model"]["name"]))
                base_row = {
                    "run_dir": run_dir.name,
                    "checkpoint_epoch": checkpoint_epoch,
                    "trial_idx": int(config.get("run", {}).get("trial_idx", 0)),
                    "dataset_preset": config["dataset"].get("preset_name", config["dataset"]["name"]),
                    "dataset_name": config["dataset"]["name"],
                    "dataset_alpha": dataset_params.get("alpha", np.nan),
                    "dataset_dim": dataset_params.get("dim", np.nan),
                    "model_name": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "model_preset": model_preset,
                    "model_alpha": model_params.get("alpha", np.nan),
                    "network_preset": config["network"]["preset_name"],
                    "train_preset": config["train"]["preset_name"],
                    "train_lr": config["train"].get("lr", np.nan),
                }

                epoch_stats = train_stats
                if "epoch" in train_stats.columns:
                    epoch_stats = train_stats[train_stats["epoch"] <= checkpoint_epoch]
                for column in ["training_loss", "grad_norm"]:
                    if column not in epoch_stats.columns:
                        continue
                    for _, stat_row in epoch_stats[["epoch", column]].dropna().iterrows():
                        scalar_rows.append(
                            {
                                **base_row,
                                "source": "train_stats",
                                "metric_name": column,
                                "value": float(stat_row[column]),
                                "eval_repeat_idx": np.nan,
                                "epoch": int(stat_row["epoch"]),
                            }
                        )

                baseline_values, baseline_warnings = compute_test_vs_test_metrics(
                    x_ref_cpu,
                    max_mmd_samples=args.max_mmd_samples,
                    mmd_device=args.device if str(args.device).startswith("cuda") else None,
                )
                for warning in baseline_warnings:
                    if warning not in warnings:
                        warnings.append(warning)
                for metric_name, metric_value in baseline_values.items():
                    scalar_rows.append(
                        {
                            **base_row,
                            "source": TEST_VS_TEST_SOURCE,
                            "metric_name": metric_name,
                            "value": metric_value,
                            "eval_repeat_idx": np.nan,
                            "epoch": np.nan,
                        }
                    )
                for eval_repeat_idx in range(int(args.n_eval_repeats)):
                    x_gen_cpu = sample_generator_in_batches(generator, len(x_ref_cpu), args.sample_batch_size)
                    metric_values, metric_warnings = compute_test_metrics(
                        x_ref_cpu,
                        x_gen_cpu,
                        max_mmd_samples=args.max_mmd_samples,
                        mmd_device=args.device if str(args.device).startswith("cuda") else None,
                    )
                    for warning in metric_warnings:
                        if warning not in warnings:
                            warnings.append(warning)
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
                    if image_class_probe is not None:
                        class_recovery_values = compute_image_class_recovery_metrics(image_class_probe, x_gen_cpu, device=args.device)
                        for metric_name in IMAGE_CLASS_RECOVERY_METRIC_NAMES:
                            scalar_rows.append(
                                {
                                    **base_row,
                                    "source": "test_metrics",
                                    "metric_name": metric_name,
                                    "value": class_recovery_values[metric_name],
                                    "eval_repeat_idx": eval_repeat_idx,
                                    "epoch": np.nan,
                                }
                            )

            pd.DataFrame(scalar_rows).to_csv(scalars_path, index=False, compression="gzip")

            summary = {
                "status": "ok",
                "run_dir": run_dir.name,
                "source_run_dir": str(run_dir.resolve()),
                "artifact_dir": str(artifact_dir.resolve()),
                "device": args.device,
                "n_checkpoints": len(checkpoint_epochs),
                "checkpoint_epochs": checkpoint_epochs,
                "n_eval_repeats": int(args.n_eval_repeats),
                "requested_n_eval_samples": None if args.n_eval_samples is None else int(args.n_eval_samples),
                "n_eval_samples": int(len(x_ref_cpu)),
                "test_split_samples": int(len(metric_ref)),
                "eval_reference_source": eval_reference_source,
                "sample_batch_size": int(args.sample_batch_size),
                "feature_dim": feature_dim,
                "image_like": bool(image_like),
                "selection_only": bool(args.selection_only),
                "selection_metric_name": None,
                "max_mmd_samples": int(args.max_mmd_samples),
                "metric_names": (
                    EVAL_METRIC_NAMES + (IMAGE_CLASS_RECOVERY_METRIC_NAMES if image_class_probe is not None else [])
                ),
                "n_scalar_rows": len(scalar_rows),
                "scalars_path": str(scalars_path),
                "warnings": warnings,
            }
            summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
            row = {
                "run_dir": run_dir.name,
                "artifact_dir": str(artifact_dir),
                "status": "ok",
                "n_scalar_rows": len(scalar_rows),
                "n_checkpoints": len(checkpoint_epochs),
            }
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
    if args.fail_on_error and n_failed:
        raise SystemExit(1)
    print("done")
