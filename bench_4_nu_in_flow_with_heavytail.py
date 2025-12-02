"""Train a flow-matching model to target a long-tail distribution using alpha-stable noise."""

# Authors: Hamza Cherkaoui

import os
import pickle
import argparse
from typing import Tuple
import tqdm
from joblib import Parallel, delayed
import numpy as np
import torch
import torch.nn as nn
from model import MLP
from utils import set_seed, get_device
from sampling import sample_student_t
from plotting import plot_heatmap
from constants import FIGURES_DIR
from training import _train
from evaluation import evaluate, fid
from loss import huber_loss


####################################################################################################
# Globals
FONTSIZE = 20
BATCH_SIZE = 512
N_EPOCHS = 250
LR = 1e-4
R = 1

def train_flow_matching_long_tail(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    nu: float,
    n_epochs: int,
    lr: float,
    device: torch.device,
    eps: float = 5e-2,
    r: float = 0.5
) -> Tuple[nn.Module, dict]:
    """Train on long-tail mixture data."""

    def step_fn(
        model: nn.Module,
        x_1: torch.Tensor,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = x_1.size(0)

        x_0 = sample_student_t(n_samples=batch_size, dim=dim, nu=nu, device=device, dtype=dtype)
        t = torch.empty(batch_size, device=device, dtype=dtype).uniform_(eps, 1 - eps)
        x_t  = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1

        return huber_loss(x_1 - x_0,  model(x_t, t))

    return _train(model, data, batch_size, n_epochs, lr, device, step_fn)


def run_single(nu_p1, nu_p0, n_samples, dim, hidden, device, dtype):
    """Single experiment runner."""

    model = MLP(dim=dim, hidden=hidden).to(device=device, dtype=dtype)

    test_base_data = sample_student_t(nu=nu_p0, n_samples=n_samples, dim=dim, dtype=dtype, device=device)
    train_ref_data = sample_student_t(nu=nu_p1, n_samples=n_samples, dim=dim, dtype=dtype, device=device)
    test_ref_data = sample_student_t(nu=nu_p1, n_samples=n_samples, dim=dim, dtype=dtype, device=device)

    model, meta = train_flow_matching_long_tail(model=model, data=train_ref_data, batch_size=BATCH_SIZE,
                                                nu=nu_p0, n_epochs=N_EPOCHS, lr=LR, r=R, device=device)

    return [(float(nu_p1), float(nu_p0)),
            evaluate(fid, test_base_data, test_ref_data, model, device, dtype),
            meta["training_loss"][-1],
            ]


####################################################################################################
# Main
if __name__ == "__main__":

    # python bench_4_nu_in_flow_with_heavytail.py --n-jobs 7

    print("[INFO] ⚙️ Setting experiment...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--results_path", type=str, default="_bench_4_cache.pkl")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.cpu)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    model = MLP(dim=args.dim, hidden=args.hidden).to(device=device, dtype=dtype)

    l_nu = np.unique(np.logspace(-1, 1, 12, dtype=float))

    if os.path.exists(args.results_path):
        print(f"[INFO] Loading existing results from {args.results_path}")
        with open(args.results_path, "rb") as f:
            saved = pickle.load(f)
        fid_results = saved.get("fid_results", {})
        loss_decrease_results = saved.get("loss_decrease_results", {})
    else:
        fid_results = {}
        loss_decrease_results = {}

    print("[INFO] 🧵 Running experiment")

    jobs = [(nu_p1, nu_p0) for nu_p1 in l_nu for nu_p0 in l_nu
            if (float(nu_p1), float(nu_p0)) not in fid_results
            or (float(nu_p1), float(nu_p0)) not in loss_decrease_results
            ]

    if len(jobs) == 0:
        print("[INFO] Nothing to do, all pairs already computed.")

    else:
        n_jobs = args.n_jobs if device.type == "cpu" else 1
        pbar = tqdm.tqdm(total=len(jobs), desc="[INFO] Main loop")
        parallel = Parallel(n_jobs=n_jobs, return_as="generator")

        l_kwargs = [{"nu_p1": nu_p1, "nu_p0": nu_p0, "n_samples": args.n_samples,
                     "dim": args.dim, "hidden": args.hidden, "device": device,
                     "dtype": dtype}
                     for (nu_p1, nu_p0) in jobs]

        for key, fid_val, loss_dec in parallel(delayed(run_single)(**kwargs) for kwargs in l_kwargs):
            fid_results[key] = fid_val
            loss_decrease_results[key] = loss_dec

            with open(args.results_path, "wb") as f:
                dump = {"fid_results": fid_results, "loss_decrease_results": loss_decrease_results}
                pickle.dump(dump, f)

            pbar.update(1)

        pbar.close()

    print(f"[INFO] Finished. Results saved to {args.results_path}")


####################################################################################################
# Plotting
    print("[INFO] 🎨 Plotting results")

    pdf_path = plot_heatmap(results=fid_results, xlabel=r"$\nu_{p_0}$", ylabel=r"$\nu_{p_\text{data}}$",
                            plot_dir=FIGURES_DIR, fontsize=FONTSIZE, suffix="fid_flow_heavytail",
                            log_scale=True, clip_percentiles=(0.0, 95.0))
    print(f"[INFO] FID heatmap plot saved at '{pdf_path}'")

    pdf_path = plot_heatmap(results=loss_decrease_results, xlabel=r"$\nu_{p_0}$", ylabel=r"$\nu_{p_\text{data}}$",
                            plot_dir=FIGURES_DIR, fontsize=FONTSIZE, suffix="loss_decrease_flow_heavytail",
                            log_scale=True, clip_percentiles=(0.0, 95.0))
    print(f"[INFO] Loss-decrease heatmap plot saved at '{pdf_path}'")
