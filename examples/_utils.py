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


def _latex_sci(x: float, digits: int = 1) -> str:
    """Format one scalar in compact LaTeX scientific notation."""
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10**exponent)
    return rf"{mantissa:.{digits}f}\,10^{{{exponent}}}"


def run_example(models, target_data_type, n_samples=10_000, exp_kwargs=None, verbose=True):
    """Train each model class on one synthetic dataset and return generated/test samples."""
    light_tailed_data = {
        "balanced_bimodal_gaussian",
        "unbalanced_bimodal_gaussian",
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
            print(f"[INFO] Running experiment on '{target_data_type}' data with '{gen_cls.__name__}' model:")

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
            print(
                f"[INFO][{i:02d}/{len(models):02d}] Evaluation: {metric_name} = {metric_value:g} "
                f"(baseline at {baseline_value:.2e}) ({elapsed:.1f} s)"
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
        title += rf"$\mathbf{{MSSLE_{{0.95}}}} \;=\; {_latex_sci(mean_mssle_95)} \;\pm\; {_latex_sci(std_mssle_95)}$"
        ax.set_title(title, fontsize=7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1.0, 0.9))
    fig.savefig(output_path)
    plt.close(fig)


def plot_path_panel(generator, fig, subplot_spec, title, n_plot_samples=500, color="tab:blue"):
    """Plot sampled 1D trajectories with start and end histograms."""
    _, trajectories = generator._sample(n_plot_samples)
    x = np.stack([state.detach().cpu().numpy().reshape(-1) for state in trajectories], axis=0).T

    _, n_local_steps = x.shape
    timesteps = np.arange(1, n_local_steps + 1)
    ymax = np.max(np.abs(x))

    inner = subplot_spec.subgridspec(1, 3, width_ratios=[1, 2.25, 1], wspace=0.0)
    ax_left = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[0, 1], sharey=ax_left)
    ax_right = fig.add_subplot(inner[0, 2], sharey=ax_left)

    for path in x:
        ax_main.plot(timesteps, path, lw=0.5, alpha=0.1, color=color)

    ax_main.set_xlim(1, n_local_steps)
    ax_main.set_xlabel("T", fontsize=7)
    ax_main.set_xticks([1, n_local_steps], ["t=1", f"T={n_local_steps - 1}"], fontsize=7)
    ax_main.set_yticks([])
    ax_main.set_ylim(-1.25 * ymax, 1.25 * ymax)
    ax_main.spines["left"].set_visible(False)
    ax_main.tick_params(axis="y", length=0)
    ax_main.set_title(title, fontsize=8)

    for values, hist_ax, invert in [(x[:, 0], ax_left, True), (x[:, -1], ax_right, False)]:
        hist_ax.hist(values, bins=50, orientation="horizontal", color=color, alpha=0.4)
        if invert:
            hist_ax.invert_xaxis()
        hist_ax.set_xticks([])
        hist_ax.spines["left"].set_visible(False)
        hist_ax.spines["bottom"].set_visible(False)
        hist_ax.tick_params(axis="both", left=False, bottom=False, labelleft=False, labelbottom=False)

    return ax_left, ax_main, ax_right
