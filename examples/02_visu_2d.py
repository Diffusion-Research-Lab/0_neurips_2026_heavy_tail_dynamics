"""Simple 1D generation illustrative example."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from genkit import (DDPMV, GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT, FlowMatchingOrigin,
                    ScoreSDEOrigin)
from genkit import DLPMEps, DLPMEpsOrigin
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import metric_on_quantile, mssle, sliced_wasserstein2
from genkit.nn import MLPModel
from genkit.training import train
from labkit.report import PRETTY_RCPARAMS


plt.rcParams.update(PRETTY_RCPARAMS)


def save_scatter_plot(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    output_path: Path,
    *,
    perc_to_plot: float = 0.99,
    pad: float = 1.05,
    alpha: float = 0.6,
    figsize: tuple[float, float] = (5, 4),
    fontsize: int = 10,
) -> None:
    """Save one 2D reference/generated scatter plot."""
    x_np = x.detach().cpu().numpy()
    x_ref_np = x_ref.detach().cpu().numpy()
    center = np.median(np.vstack([x_np, x_ref_np]), axis=0)
    half_xy = np.quantile(np.abs(np.vstack([x_np, x_ref_np]) - center), perc_to_plot, axis=0)
    half = float(np.max(half_xy))

    plt.figure(figsize=figsize)
    plt.scatter(x_ref_np[:, 0], x_ref_np[:, 1], s=6, label="Target", alpha=alpha)
    plt.scatter(x_np[:, 0], x_np[:, 1], s=6, label="Generated", alpha=alpha)
    plt.xlim(center[0] - pad * half, center[0] + pad * half)
    plt.ylim(center[1] - pad * half, center[1] + pad * half)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(fontsize=fontsize)
    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_models(models, target_data_type, *, n_samples: int, exp_kwargs: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Train models on one synthetic dataset and return generated/reference samples."""
    light_tailed_data = {
        "balanced_bimodal_gaussian",
        "unbalanced_bimodal_gaussian",
        "unbalanced_highdim_gaussian_mixture",
        "gaussian",
        "checker",
        "spiral",
    }
    dim = exp_kwargs.get("dim", 2)
    extra_data_kwargs = exp_kwargs.get("extra_data_kwargs", {})
    extra_gen_kwargs = exp_kwargs.get("extra_gen_kwargs", {})
    n_steps = exp_kwargs.get("n_steps", 250)
    batch_size = exp_kwargs.get("batch_size", 1024)
    n_epochs = exp_kwargs.get("n_epochs", 100)
    lr = exp_kwargs.get("lr", 1e-3)
    width = exp_kwargs.get("width", 32)
    depth = exp_kwargs.get("depth", 2)
    num_workers = exp_kwargs.get("num_workers", 0)
    device = exp_kwargs.get("device", "cpu")
    fdtype = exp_kwargs.get("fdtype", torch.float32)
    idtype = exp_kwargs.get("idtype", torch.int32)

    x_train, _, x_test = fetch_synthetic_data(
        target_data_type,
        n_samples=n_samples,
        dim=dim,
        device=device,
        dtype=fdtype,
        **extra_data_kwargs,
    )
    train_kwargs = {
        "target_data": x_train,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "device": device,
        "num_workers": num_workers,
    }
    if target_data_type in light_tailed_data:
        baseline_value = sliced_wasserstein2(x_test, x_train)
        metric_name = "Wasserstein-dist"
    else:
        baseline_value = metric_on_quantile(mssle, x_test, x_train, xi=0.95)
        metric_name = "MSSLE(95)"

    results = {}
    for i, gen_cls in enumerate(models, start=1):
        print(
            f"[TRAIN] {gen_cls.__name__}"
            f" | dataset={target_data_type}"
            f" | index={i:02d}/{len(models):02d}"
            f" | epochs={n_epochs}"
            f" | batch_size={batch_size}"
            f" | lr={lr:.1e}"
        )
        net = MLPModel(dim=dim, width=width, depth=depth).to(device=device, dtype=fdtype)
        generator = gen_cls(
            net=net,
            dim=dim,
            fdtype=fdtype,
            idtype=idtype,
            device=device,
            n_steps=n_steps,
            **extra_gen_kwargs,
        )
        train(generative_model=generator, **train_kwargs)
        x_test_gen = generator.sample(n_samples=n_samples)
        if target_data_type in light_tailed_data:
            metric_value = sliced_wasserstein2(x_test, x_test_gen)
        else:
            metric_value = metric_on_quantile(mssle, x_test, x_test_gen, xi=0.95)
        print(
            f"[EVAL] {gen_cls.__name__}"
            f" | index={i:02d}/{len(models):02d}"
            f" | metric={metric_name}"
            f" | value={metric_value:.2e}"
            f" | baseline={baseline_value:.2e}"
        )
        results[gen_cls.__name__] = (x_test_gen, x_test)
    return results


if __name__ == "__main__":

    figures_dir = "_figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--blank", action="store_false", help="CI helper.")
    parser.add_argument("--data", type=str, default='spiral', help="Data distribution.")
    args = parser.parse_args()

    light_tailed_data = ["balanced_bimodal_gaussian",
                         "unbalanced_bimodal_gaussian",
                         "unbalanced_highdim_gaussian_mixture",
                         "gaussian",
                         "checker",
                         "spiral",
                         ]
    light_tailed_models = [DDPMV,
                           GaussianFlowDDPM,
                           GaussianFlowLinear,
                           GaussianFlowOT,
                           FlowMatchingOrigin,
                           ScoreSDEOrigin,
                           ]

    if args.data in light_tailed_data:
        models = light_tailed_models
    else:
        models = [DLPMEps,
                  DLPMEpsOrigin,
                  ]

    exp_kwargs = dict(extra_data_kwargs=dict(),
                      n_steps=300,
                      batch_size=1024,
                      n_epochs=250,
                      lr=1e-3,
                      width=64,
                      depth=3,
                      extra_gen_kwargs=dict(),
                      device="cuda" if torch.cuda.is_available() else "cpu",
                      fdtype=torch.float32,
                      idtype=torch.int32,
                      )

    print(
        "2D sample visualization"
        f" | dataset={args.data}"
        f" | device={exp_kwargs['device']}"
        f" | n_samples={10_000 if args.blank else 100}"
        f" | n_steps={exp_kwargs['n_steps']}"
        f" | batch_size={exp_kwargs['batch_size']}"
        f" | n_epochs={exp_kwargs['n_epochs']}"
    )

    results = run_models(
        models,
        target_data_type=args.data,
        n_samples=10_000 if args.blank else 100,
        exp_kwargs=exp_kwargs,
    )

    for i, (name, (x_gen, x_ref)) in enumerate(results.items()):
        save_scatter_plot(
            x=x_gen,
            x_ref=x_ref,
            output_path=figures_dir / f"{args.data}_{name}_2d_scatter.pdf",
            fontsize=10,
            alpha=0.6,
        )

    print(f"saved={figures_dir}")
