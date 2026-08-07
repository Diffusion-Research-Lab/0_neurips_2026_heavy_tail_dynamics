#!/usr/bin/env python
"""Sample and save visualization grids for the ImageNet-LT visualization benchmark."""

import argparse
import importlib
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import pandas as pd  # noqa
import torch  # noqa
from toolkit.utils import latest_config_batch_dir  # noqa

_main = importlib.import_module("benchmarks.01_main")
build_model = _main.build_model
build_network = _main.build_network


MODEL_ORDER = [
    "gaussian_flow_linear_euler",
    "gaussian_flow_linear_heun",
    "ddpm_v_ddpm",
    "ddpm_v_ddim",
    "dlpm_eps_a17",
    "dlpm_eps_a19",
    "tedm_origin_nu21",
    "tedm_origin_nu30",
]
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
VIZ_DATASET_NAME = "imagenet_lt_96"
VIZ_LABEL = "ImageNet-LT 96"


def checkpoint_dtype(checkpoint: dict[str, Any]) -> torch.dtype:
    dtype_spec = checkpoint.get("model_init", {}).get("dtype", torch.float32)
    return getattr(torch, str(dtype_spec).split(".")[-1], torch.float32)


def run_dirs_for_config(artifact_root: Path, config_path: Path) -> list[Path]:
    batch_dir = latest_config_batch_dir(artifact_root, config_path)
    return sorted(path for path in batch_dir.iterdir() if path.is_dir() and (path / "checkpoint.pt").is_file())


def build_generator_from_checkpoint(run_dir: Path, device: torch.device):
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    dtype = checkpoint_dtype(checkpoint)
    model_init = checkpoint["model_init"]
    train_shape = tuple(int(axis) for axis in model_init["train_shape"])
    shape_ref = torch.empty((1, *train_shape[1:]), dtype=dtype)

    network_cfg = model_init["network"]
    model_cfg = model_init["model"]
    net, _ = build_network(network_cfg, shape_ref)
    net.load_state_dict(checkpoint["network_state_dict"])
    net = net.to(device=device, dtype=dtype)
    generator, _ = build_model(model_cfg, net, shape_ref, dtype=dtype, device=device)
    generator._net.eval()
    return generator


def sample_in_batches(generator: Any, n_samples: int, batch_size: int) -> torch.Tensor:
    chunks = []
    remaining = int(n_samples)
    while remaining > 0:
        n_batch = min(int(batch_size), remaining)
        with torch.no_grad():
            chunks.append(generator.sample(n_samples=n_batch).detach().cpu())
        remaining -= n_batch
    return torch.cat(chunks, dim=0)


def prepare_images(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim != 4:
        raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(x.shape)}.")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    if mode == "clamp":
        return x.clamp(0.0, 1.0)
    if mode == "per-image":
        flat = x.flatten(start_dim=1)
        lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    raise ValueError(f"Unknown image normalization mode {mode!r}.")


def save_grid(images: torch.Tensor, output_path: Path, title: str, normalize: str, n_cols: int | None = None) -> None:
    images = prepare_images(images, mode=normalize)
    n_images = int(images.shape[0])
    n_cols = int(n_cols or math.ceil(math.sqrt(n_images)))
    n_rows = int(math.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), squeeze=False)
    for index, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if index < n_images:
            ax.imshow(images[index].permute(1, 2, 0).numpy())
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=PROJECT_ROOT / "benchmarks" / "configs" / "viz" / VIZ_DATASET_NAME)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "benchmarks" / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "figures" / f"{VIZ_DATASET_NAME}_viz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--trial-index", type=int, default=0, help="Which completed trial/run to visualize for each model.")
    parser.add_argument("--normalize", choices=["clamp", "per-image"], default="clamp")
    args = parser.parse_args()

    device = torch.device(args.device)
    records = []
    for model_name in MODEL_ORDER:
        config_path = args.config_root / f"{model_name}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing visualization config: {config_path}")
        run_dirs = run_dirs_for_config(args.artifact_root, config_path)
        if not run_dirs:
            raise FileNotFoundError(f"No completed runs found for {config_path}")
        run_dir = run_dirs[min(int(args.trial_index), len(run_dirs) - 1)]
        generator = build_generator_from_checkpoint(run_dir, device=device)
        samples = sample_in_batches(generator, n_samples=args.n_samples, batch_size=args.sample_batch_size)
        output_path = args.output_dir / f"{VIZ_DATASET_NAME}__{model_name}.png"
        save_grid(samples, output_path, title=f"{VIZ_LABEL} - {MODEL_LABELS.get(model_name, model_name)}", normalize=args.normalize)
        records.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "output_path": str(output_path),
                "n_samples": int(args.n_samples),
                "sample_shape": tuple(samples.shape),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"Wrote {len(records)} visualization grids under {args.output_dir}")


if __name__ == "__main__":
    main()
