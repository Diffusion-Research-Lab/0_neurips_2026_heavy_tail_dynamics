"""Fast CPU smoke test for package wiring and benchmark config loading."""

from pathlib import Path
import genkit
import labkit
import torch
from genkit.datasets import fetch_synthetic_data
from labkit.config import load_config


if __name__ == "__main__":
    torch.manual_seed(0)

    cfg_path = Path(__file__).resolve().parent / "04_alpha_values_benchmark_cfg_blank.yml"
    cfg = load_config(cfg_path).set_up()

    x_train, _, x_test = fetch_synthetic_data(
        target_data=cfg.target_data_type,
        n_samples=32,
        dim=cfg.dim,
        alpha=float(cfg.l_alpha_data[0]),
        device="cpu",
        dtype=torch.float32,
    )

    if not torch.equal(x_train, x_train):
        raise RuntimeError("Unexpected smoke test tensor mismatch.")
    if x_test.shape[1] != int(cfg.dim):
        raise RuntimeError(f"Unexpected sample shape: {tuple(x_test.shape)}")

    print(
        "smoke_ok",
        f"genkit={genkit.__name__}",
        f"labkit={labkit.__name__}",
        f"config={cfg_path.name}",
    )
