"""Train a flow-matching model to target a long-tail distribution using alpha-stable noise."""

# Authors: Hamza Cherkaoui

import argparse
import logging
from typing import Tuple
import torch
import torch.nn as nn
from model import MLP
from utils import set_seed, get_device
from sampling import sample_bimodal_gaussian, sample_gaussian
from plotting import plot_training_loss
from constants import FIGURES_DIR, FONTSIZE
from training import _train


def train_flow_matching_gaussian(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: torch.device,
    eps: float = 1e-2,
) -> Tuple[nn.Module, dict]:
    """Train on long-tail data (Gaussian base)."""

    loss_fn = nn.MSELoss()

    def step_fn(
        model: nn.Module,
        x_1: torch.Tensor,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = x_1.size(0)

        x_0 = sample_gaussian(n_samples=batch_size, dim=dim, device=device, dtype=dtype)

        t = torch.empty(batch_size, device=device, dtype=dtype).uniform_(eps, 1 - eps)

        g = sample_gaussian(n_samples=batch_size, dim=dim, device=device, dtype=dtype)
        sigma_g = torch.sqrt(t * (1 - t)).unsqueeze(-1)
        d_sigma_g_dt = (0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))).unsqueeze(-1)

        v_t = (x_1 - x_0) + d_sigma_g_dt * g
        x_t = x_0 + t.unsqueeze(-1) * (x_1 - x_0) + sigma_g * g

        return loss_fn(v_t, model(x_t, t))

    return _train(model, data, batch_size, n_epochs, lr, device, step_fn)


if __name__ == "__main__":

    print("[INFO] ⚙️ Setting experiment...")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.cpu)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    model = MLP(dim=args.dim, hidden=args.hidden).to(device=device, dtype=dtype)

    print("[INFO] 🧮 Generating dataset")
    data = sample_bimodal_gaussian(n_samples=args.n_samples,
                                   dim=args.dim,
                                   device=device,
                                   )

    print("[INFO] 🧵 Running experiment")
    model, meta = train_flow_matching_gaussian(model=model,
                                               data=data,
                                               batch_size=args.batch_size,
                                               n_epochs=args.n_epochs,
                                               lr=args.lr,
                                               device=device,
                                               )

    print("[INFO] 🎨 Plotting results")
    pdf_path = plot_training_loss(l_loss=meta['training_loss'],
                                  plot_dir=FIGURES_DIR,
                                  xlogscale=False,
                                  ylogscale=False,
                                  fontsize=FONTSIZE,
                                  suffix="flowmatching_gaussian",
                                  )
    logging.info(f"Training loss plot saved at '{pdf_path}'")
