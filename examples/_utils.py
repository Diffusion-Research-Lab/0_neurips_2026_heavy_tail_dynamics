"""Shared helpers for the example scripts."""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from genkit._sampling import sample_scaled_isotropic_alpha_stable
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import metric_on_quantile, mssle, sliced_wasserstein2
from genkit.nn import MLPModel
from genkit.training import train
from labkit.report import to_latex_sci


def _format_fields(**kwargs) -> str:
    """Format one compact key=value field list for example prints."""
    return " | ".join(f"{key}={value}" for key, value in kwargs.items())


def print_start(name: str, **kwargs) -> None:
    """Print one standardized example start line."""
    fields = _format_fields(**kwargs)
    print(f"[START] {name}" + (f" | {fields}" if fields else ""))


def print_model_step(tag: str, model_name: str, **kwargs) -> None:
    """Print one standardized per-model progress line."""
    fields = _format_fields(**kwargs)
    print(f"[{tag}] {model_name}" + (f" | {fields}" if fields else ""))


def print_done(**kwargs) -> None:
    """Print one standardized example completion line."""
    fields = _format_fields(**kwargs)
    print(f"[DONE] {fields}" if fields else "[DONE]")


def _format_tick(v: float, dec: int = 2, tol: float = 1e-6) -> str:
    """Format tick value compactly for example figures."""
    if np.isfinite(v) and abs(v - round(v)) < tol:
        return str(int(round(v)))
    return f"{v:.{dec}g}"


def plot_scatter(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    plot_dir: str | Path,
    *,
    perc_to_plot: float = 0.99,
    pad: float = 1.05,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (5, 4),
    fontsize: int = 18,
    suffix: str = "experiment",
) -> str:
    """Save one 2D reference/generated scatter plot for examples."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"{suffix}_2d_scatter.pdf"

    x = x.detach().cpu().numpy()
    x_ref = x_ref.detach().cpu().numpy()
    center = np.median(np.vstack([x, x_ref]), axis=0)
    half_xy = np.quantile(np.abs(np.vstack([x, x_ref]) - center), perc_to_plot, axis=0)
    half = float(np.max(half_xy))

    plt.figure(figsize=figsize)
    plt.scatter(x_ref[:, 0], x_ref[:, 1], s=6, label="Target", alpha=alpha)
    plt.scatter(x[:, 0], x[:, 1], s=6, label="Generated", alpha=alpha)
    plt.xlim(center[0] - pad * half, center[0] + pad * half)
    plt.ylim(center[1] - pad * half, center[1] + pad * half)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(fontsize=fontsize)
    plt.tight_layout(pad=0.8)
    plt.savefig(pdf_path, dpi=300)
    plt.close()

    return str(pdf_path)


def run_example(models, target_data_type, n_samples=10_000, exp_kwargs=None, verbose=True):
    """Train each model class on one synthetic dataset and return generated/test samples."""
    light_tailed_data = {
        "balanced_bimodal_gaussian",
        "unbalanced_bimodal_gaussian",
        "unbalanced_highdim_gaussian_mixture",
        "gaussian",
        "checker",
        "spiral",
    }
    exp_kwargs = {} if exp_kwargs is None else dict(exp_kwargs)

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
        if verbose:
            print_model_step(
                "TRAIN",
                gen_cls.__name__,
                dataset=target_data_type,
                index=f"{i:02d}/{len(models):02d}",
                epochs=n_epochs,
                batch_size=batch_size,
                lr=f"{lr:.1e}",
            )

        t0 = time.perf_counter()
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

        if verbose:
            elapsed = time.perf_counter() - t0
            print_model_step(
                "EVAL",
                gen_cls.__name__,
                index=f"{i:02d}/{len(models):02d}",
                metric=metric_name,
                value=f"{metric_value:.2e}",
                baseline=f"{baseline_value:.2e}",
                elapsed=f"{elapsed:.1f}s",
            )

        results[gen_cls.__name__] = (x_test_gen, x_test)

    return results


def plot_generated_samples(results, model_specs, data_kwargs, n_trials: int, output_path: Path) -> None:
    """Plot one 2D reference/generated scatter panel per model and save the figure."""
    trial_to_plot = 0
    lim = 20
    n_samples = 10_000

    ref_kwargs = dict(data_kwargs)
    ref_kwargs["n_samples"] = n_samples
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_samples = [sample_scaled_isotropic_alpha_stable(**ref_kwargs) for _ in range(n_trials)]
    fig, axes = plt.subplots(1, len(model_specs), figsize=(2.0 * len(model_specs), 2.5), squeeze=False)

    for i, (spec, res) in enumerate(zip(model_specs, results, strict=True)):
        gen_samples = [generator.sample(n_samples) for generator in res["generators"]]
        all_mssle_95 = [metric_on_quantile(mssle, x_gen, x_ref, xi=0.95) for x_gen, x_ref in zip(gen_samples, ref_samples)]
        mean_mssle_95 = np.mean(all_mssle_95)
        std_mssle_95 = np.std(all_mssle_95)

        x_ref = ref_samples[trial_to_plot].detach().cpu()
        x_gen = gen_samples[trial_to_plot].detach().cpu()
        ax = axes[0, i]
        ax.scatter(x_ref[:, 0], x_ref[:, 1], s=3.0, color="tab:blue", alpha=0.5)
        ax.scatter(x_gen[:, 0], x_gen[:, 1], s=3.0, color="tab:orange", alpha=0.5)
        ax.text(0.48, 0.01, "Ref.", color="tab:blue", ha="right", va="bottom", transform=ax.transAxes, fontsize=8)
        ax.text(0.52, 0.01, "Gen.", color="tab:orange", ha="left", va="bottom", transform=ax.transAxes, fontsize=8)
        ax.text(0.5, -0.06, f"(trial {trial_to_plot + 1})", ha="center", va="bottom", transform=ax.transAxes, fontsize=8)

        title = rf"$\mathbf{{{spec['name']}}}$" + "\n\n"
        title += rf"$\mathbf{{MSSLE_{{0.95}}}} \;=\; {to_latex_sci(mean_mssle_95, digits=1)} \;\pm\; {to_latex_sci(std_mssle_95, digits=1)}$"
        ax.set_title(title, fontsize=7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1.0, 0.9))
    fig.savefig(output_path)
    plt.close(fig)
