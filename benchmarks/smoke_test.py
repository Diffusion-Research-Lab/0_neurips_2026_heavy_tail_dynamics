"""Fast CPU smoke test for package wiring and benchmark config loading."""

from pathlib import Path
import torch
import cauda
import labkit
from cauda.datasets import fetch_synthetic_data
from cauda.metrics import mse
from labkit.config import load_config


def main() -> None:
    torch.manual_seed(0)

    cfg_path = Path(__file__).resolve().parent / "config_blank" / "bench_1_config_blank.yml"
    cfg = load_config(cfg_path).set_up()

    x_train, _, x_test = fetch_synthetic_data(
        target_data=cfg.target_data_type,
        n_samples=32,
        dim=cfg.dim,
        alpha=float(cfg.alpha_data),
        device="cpu",
        dtype=torch.float32,
    )

    score = float(mse(x_train, x_train).item())
    if score != 0.0:
        raise RuntimeError(f"Unexpected smoke metric value: {score}")
    if x_test.shape[1] != int(cfg.dim):
        raise RuntimeError(f"Unexpected sample shape: {tuple(x_test.shape)}")

    print(
        "smoke_ok",
        f"cauda={cauda.__name__}",
        f"labkit={labkit.__name__}",
        f"config={cfg_path.name}",
    )


if __name__ == "__main__":
    main()
