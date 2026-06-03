"""Batch evaluation entrypoint for saved benchmark runs."""

import argparse
import copy
import importlib
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

import numpy as np                                                                                               # noqa
import pandas as pd                                                                                              # noqa
import torch                                                                                                     # noqa
from torch import nn                                                                                             # noqa
import yaml                                                                                                      # noqa
from benchmarks._real_data_cache import load_preprocessed_real_dataset_metadata, real_dataset_cache_key           # noqa
from genkit.metrics import mmd_rbf, tail_coverage_error                                                           # noqa


_main = importlib.import_module("benchmarks.01_main")
build_dataset = _main.build_dataset
build_model = _main.build_model
build_network = _main.build_network
setup_logging = _main.setup_logging


EVAL_METRIC_NAMES = [
    "MMD_RBF",
    "TCE(90%)",
    "TCE(95%)",
    "TCE(99%)",
    "TCE(99.9%)",
]
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
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--max-mmd-samples", type=int, default=2048)
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


def encode_class_labels(labels: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """Map original class ids to classifier output indices; unknown ids become -1."""
    mapping = {int(class_id): index for index, class_id in enumerate(class_ids.tolist())}
    return torch.tensor([mapping.get(int(value), -1) for value in labels.tolist()], dtype=torch.long)


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


def class_histogram(labels: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """Return counts over class_ids for a vector of original class labels."""
    encoded = encode_class_labels(labels.detach().cpu().long(), class_ids.detach().cpu().long())
    valid = encoded.ge(0)
    return torch.bincount(encoded[valid], minlength=int(class_ids.numel())).float()


def class_histogram_tv(test_probs: torch.Tensor, gen_counts: torch.Tensor, n_generated: int) -> float:
    """Total variation between real test class probabilities and generated class predictions."""
    if int(n_generated) <= 0:
        return float("nan")
    gen_probs = gen_counts.float() / float(n_generated)
    outside_mass = max(0.0, 1.0 - float(gen_probs.sum().item()))
    return float(0.5 * (torch.abs(test_probs.float() - gen_probs).sum().item() + outside_mass))


def train_image_class_probe(
    x_train_cpu: torch.Tensor,
    train_labels: torch.Tensor,
    x_test_cpu: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    device: str,
) -> dict[str, Any]:
    """Train a small supervised classifier probe on the real image train split."""
    if x_train_cpu.ndim != 4:
        raise ValueError(f"classifier probe expects image tensors with shape (N,C,H,W), got {tuple(x_train_cpu.shape)}")
    if len(train_labels) != len(x_train_cpu):
        raise ValueError(f"train label count {len(train_labels)} does not match x_train count {len(x_train_cpu)}")
    if len(test_labels) != len(x_test_cpu):
        raise ValueError(f"test label count {len(test_labels)} does not match x_test count {len(x_test_cpu)}")

    cuda_devices: list[int] = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        device_obj = torch.device(device)
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

        y_train = encode_class_labels(train_labels, class_ids)
        valid_train = y_train.ge(0)
        x_train_cpu = x_train_cpu[valid_train]
        y_train = y_train[valid_train]
        if len(y_train) == 0:
            raise ValueError("classifier probe has no labeled train samples")

        model = ImageClassProbeNet(int(x_train_cpu.shape[1]), int(class_ids.numel())).to(device=device)
        counts = torch.bincount(y_train, minlength=int(class_ids.numel())).float()
        class_weights = (counts.sum() / counts.clamp_min(1.0)).sqrt()
        class_weights = class_weights / class_weights.mean().clamp_min(1.0e-12)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=IMAGE_CLASSIFIER_LR, weight_decay=1.0e-4)
        batch_size = min(int(IMAGE_CLASSIFIER_BATCH_SIZE), len(x_train_cpu))

        model.train()
        for _ in range(int(IMAGE_CLASSIFIER_EPOCHS)):
            perm = torch.randperm(len(x_train_cpu))
            for start in range(0, len(x_train_cpu), batch_size):
                idx = perm[start: start + batch_size]
                x = x_train_cpu[idx].to(device=device, dtype=torch.float32)
                y = y_train[idx].to(device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()

        test_pred = predict_image_class_probe({"model": model, "class_ids": class_ids}, x_test_cpu, device=device)
        known_test = encode_class_labels(test_labels, class_ids).ge(0)
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

        test_hist = class_histogram(known_test_labels, eligible_classes)
        return {
            "model": model.cpu(),
            "class_ids": class_ids.cpu(),
            "eligible_classes": eligible_classes.cpu(),
            "test_probs": (test_hist / test_hist.sum().clamp_min(1.0)).cpu(),
            "test_acc": test_acc,
            "n_test_classes": int(eligible_classes.numel()),
        }


def image_class_probe_cache_path(artifact_root: Path, dataset_name: str, dataset_cfg: dict[str, Any], dtype: torch.dtype) -> Path:
    """Return the shared classifier-probe cache path for one image dataset request."""
    cache_key = real_dataset_cache_key(dataset_cfg, dtype)
    return artifact_root / "class_probes" / f"{dataset_name}_{cache_key}_v{IMAGE_CLASSIFIER_VERSION}.pt"


def save_image_class_probe_cache(path: Path, probe: dict[str, Any]) -> None:
    """Atomically persist a trained classifier probe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": IMAGE_CLASSIFIER_VERSION,
        "class_ids": probe["class_ids"].cpu(),
        "eligible_classes": probe["eligible_classes"].cpu(),
        "test_probs": probe["test_probs"].cpu(),
        "test_acc": float(probe["test_acc"]),
        "n_test_classes": int(probe["n_test_classes"]),
        "state_dict": probe["model"].cpu().state_dict(),
        "in_channels": int(probe["model"].net[0].in_channels),
        "num_classes": int(probe["class_ids"].numel()),
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_image_class_probe_cache(path: Path) -> dict[str, Any]:
    """Load a trained classifier probe from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != IMAGE_CLASSIFIER_VERSION:
        raise ValueError(f"classifier probe cache version mismatch: {path}")
    model = ImageClassProbeNet(int(payload["in_channels"]), int(payload["num_classes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "class_ids": payload["class_ids"].long(),
        "eligible_classes": payload["eligible_classes"].long(),
        "test_probs": payload["test_probs"].float(),
        "test_acc": float(payload["test_acc"]),
        "n_test_classes": int(payload["n_test_classes"]),
    }


def compute_image_class_recovery_metrics(probe: dict[str, Any], x_gen_cpu: torch.Tensor, *, device: str) -> dict[str, float]:
    """Compute class recovery diagnostics from generated image samples."""
    predicted_classes = predict_image_class_probe(probe, x_gen_cpu, device=device)
    generated_classes = set(int(value) for value in torch.unique(predicted_classes).tolist())
    eligible_class_tensor = probe["eligible_classes"].detach().cpu().long()
    eligible_classes = set(int(value) for value in eligible_class_tensor.tolist())
    recovered = len(generated_classes & eligible_classes)
    n_test_classes = int(probe["n_test_classes"])
    gen_counts = class_histogram(predicted_classes, eligible_class_tensor)
    return {
        "CLASS_RECOVERY_INDEX": float(recovered / n_test_classes) if n_test_classes else float("nan"),
        "CLASS_HIST_TV": class_histogram_tv(probe["test_probs"], gen_counts, len(predicted_classes)),
        "CLASS_PROBE_TEST_ACC": float(probe["test_acc"]),
        "N_TEST_CLASSES": float(n_test_classes),
        "N_GEN_CLASSES": float(len(generated_classes)),
    }


def prepare_image_class_recovery_probe(reference_run_dir: Path, *, device: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Prepare one trained image classifier probe for CIFAR100-LT or ImageNet-LT batches."""
    config = yaml.safe_load((reference_run_dir / "config.yaml").read_text())
    dataset_cfg = config["dataset"]
    dataset_name = str(dataset_cfg.get("name", ""))
    if dataset_name not in IMAGE_CLASS_RECOVERY_DATASETS:
        return None, []

    try:
        final_checkpoint = torch.load(reference_run_dir / "checkpoint.pt", map_location="cpu")
        dtype = checkpoint_dtype(config, final_checkpoint)
        artifact_root = reference_run_dir.parent.parent
        cache_path = image_class_probe_cache_path(artifact_root, dataset_name, dataset_cfg, dtype)
        if cache_path.is_file():
            return load_image_class_probe_cache(cache_path), []

        x_train, _, x_test = build_dataset(dataset_cfg, dtype=dtype, device="cpu")
        metadata = load_preprocessed_real_dataset_metadata(dataset_cfg, dtype=dtype)
        train_labels = metadata_class_labels(metadata, "train")
        test_labels = metadata_class_labels(metadata, "test")
        if train_labels is None or test_labels is None:
            raise ValueError("single-label class metadata is missing")
        probe = train_image_class_probe(
            x_train.detach().cpu(),
            train_labels,
            x_test.detach().cpu(),
            test_labels,
            device=device,
        )
        save_image_class_probe_cache(cache_path, probe)
        return probe, []
    except Exception as exc:
        return None, [f"class_recovery_probe_skipped: {type(exc).__name__}: {exc}"]


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
    image_class_probe: dict[str, Any] | None,
    image_class_probe_warnings: list[str],
    device: str,
    n_eval_samples: int,
    n_eval_repeats: int,
    sample_batch_size: int,
    max_mmd_samples: int,
    selection_only: bool,
    selection_split: str,
    selection_repeats: int,
    selection_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate one saved benchmark run and write per-run artifacts."""
    artifact_dir = artifact_batch_dir / run_dir.name
    scalars_path = artifact_dir / "scalars.csv.gz"
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
    warnings: list[str] = list(image_class_probe_warnings)

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
                max_mmd_samples=max_mmd_samples,
                mmd_device=device if str(device).startswith("cuda") else None,
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
            if image_class_probe is not None:
                class_recovery_values = compute_image_class_recovery_metrics(image_class_probe, x_gen_cpu, device=device)
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
        "device": device,
        "n_checkpoints": len(checkpoint_epochs),
        "checkpoint_epochs": checkpoint_epochs,
        "n_eval_repeats": int(n_eval_repeats),
        "n_eval_samples": int(n_eval_samples),
        "sample_batch_size": int(sample_batch_size),
        "feature_dim": feature_dim,
        "image_like": bool(image_like),
        "selection_only": bool(selection_only),
        "selection_split": selection_split,
        "selection_metric_name": selection_metric_name if selection_only else None,
        "selection_repeats": int(selection_repeats),
        "selection_batch_size": int(selection_batch_size),
        "max_mmd_samples": int(max_mmd_samples),
        "metric_names": (
            [selection_metric_name]
            if selection_only
            else EVAL_METRIC_NAMES + (IMAGE_CLASS_RECOVERY_METRIC_NAMES if image_class_probe is not None else [])
        ),
        "n_scalar_rows": len(scalar_rows),
        "scalars_path": str(scalars_path),
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


if __name__ == "__main__":
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
    if args.selection_only:
        image_class_probe, image_class_probe_warnings = None, []
    else:
        image_class_probe, image_class_probe_warnings = prepare_image_class_recovery_probe(runs[0], device=args.device)
    manifest_rows = []
    for run_index, run_dir in enumerate(runs, start=1):
        if (run_index - 1) % args.shard_count != args.shard_index:
            continue
        try:
            row = evaluate_one_run(
                run_dir,
                artifact_batch_dir=artifact_batch_dir,
                image_class_probe=image_class_probe,
                image_class_probe_warnings=image_class_probe_warnings,
                device=args.device,
                n_eval_samples=args.n_eval_samples,
                n_eval_repeats=args.n_eval_repeats,
                sample_batch_size=args.sample_batch_size,
                max_mmd_samples=args.max_mmd_samples,
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
