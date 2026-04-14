"""Compare DLPM variants on 2D alpha-stable data."""

import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from genkit import DLPMEpsOrigin
from genkit.datasets import fetch_synthetic_data
from genkit.nn import MLPModel
from genkit.training import train
from genkit.visitor import CoreMetricsVisitor
from labkit.report import PRETTY_RCPARAMS, to_latex_sci
from genkit._sampling import sample_scaled_isotropic_alpha_stable
from genkit.metrics import metric_on_quantile, mssle


plt.rcParams.update(PRETTY_RCPARAMS)


def plot_dlpm_samples(results, model_specs, data_kwargs, n_trials: int, output_path: Path) -> None:
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
        mean_mssle_95 = sum(all_mssle_95) / len(all_mssle_95)
        std_mssle_95 = float(torch.tensor(all_mssle_95).std(unbiased=False).item())

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--blank", action="store_false", help="CI helper.")
    args = parser.parse_args()

    figures_dir = Path("_figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    dim = 2
    alpha_data = 1.6
    n_train_samples = 10_000 if args.blank else 100
    n_steps = 128
    batch_size = 1024
    n_epochs = 1024
    lr = 1e-3
    n_trials = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fdtype = torch.float32
    idtype = torch.int32

    model_specs = [
        {"name": "DLPM(1.7)", "cls": DLPMEpsOrigin, "gen_kwargs": {"alpha": 1.6}},
        {"name": "DLPM(2.0)", "cls": DLPMEpsOrigin, "gen_kwargs": {"alpha": 2.0}},
    ]

    data_kwargs = dict(alpha=alpha_data, n_samples=n_train_samples, dim=dim, device=device, dtype=fdtype)
    x_train, _, _ = fetch_synthetic_data(target_data="alpha_stable", **data_kwargs)

    print(
        "Shariatan et al comparison"
        f" | device={device}"
        f" | alpha={alpha_data}"
        f" | n_samples={n_train_samples}"
        f" | n_steps={n_steps}"
        f" | batch_size={batch_size}"
        f" | n_epochs={n_epochs}"
        f" | n_trials={n_trials}"
    )

    train_kwargs = dict(target_data=x_train, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                        device=device, visitors=[CoreMetricsVisitor()])
    net_kwargs = dict(width=32, depth=3, time_dim=16, dropout=0.0, use_norm=True)

    results = []
    total_runs = n_trials * len(model_specs)
    run_idx = 0

    for spec in model_specs:
        generators = []
        diagnostics = []

        for _ in range(n_trials):
            run_idx += 1
            net = MLPModel(dim=dim, **net_kwargs).to(device=device, dtype=fdtype)
            generator = spec["cls"](net=net, dim=dim, n_steps=n_steps, device=device, fdtype=fdtype, idtype=idtype, **spec["gen_kwargs"])

            t0 = time.time()
            print(
                f"[TRAIN] {spec['name']}"
                f" | index={run_idx:02d}/{total_runs:02d}"
                f" | epochs={n_epochs}"
                f" | batch_size={batch_size}"
                f" | lr={lr:.1e}"
            )
            generator, diagnostic = train(generative_model=generator, **train_kwargs)
            print(f"[EVAL] {spec['name']} | index={run_idx:02d}/{total_runs:02d} | elapsed={time.time() - t0:.1f}s")

            generators.append(generator)
            diagnostics.append(diagnostic)

        results.append({"generators": generators, "diagnostics": diagnostics})

    plot_dlpm_samples(results, model_specs, data_kwargs, n_trials, figures_dir / "shariatan_et_al_samples.pdf")

    print(f"saved={figures_dir}")
