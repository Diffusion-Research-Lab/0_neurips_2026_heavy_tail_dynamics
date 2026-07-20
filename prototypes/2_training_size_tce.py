import logging
from pathlib import Path
import pandas as pd
import torch
from genkit.diffusion import DDPMV, DLPMEps
from genkit.training import train
from _constants import ALPHA, DIM, N_MMD, N_STEPS, N_TEST, N_TRIALS, TCE_COLUMNS, TCE_TAIL_PROBS
from _utils import add_test_vs_true_sample, evaluate_model, load_alpha_stable, make_net, make_train_kwargs, setup

import matplotlib.pyplot as plt


log_path = Path(__file__).with_name("logs") / f"{Path(__file__).stem}.log"
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


########################################################################################################################
# Additional constants and functions

TRAIN_SIZES = [1_000, 3_000, 10_000, 20_000, 30_000]


def save_training_size_figure(rows, filename):
    results = pd.DataFrame(rows)
    reference = results[results["model"] == "test vs true sample"][TCE_COLUMNS].mean()
    model_results = results[results["model"] != "test vs true sample"]
    grouped = model_results.groupby(["model", "n_train"], sort=False)
    mean_results = grouped[["mmd_rbf", *TCE_COLUMNS]].mean().reset_index()

    quantiles = 1.0 - TCE_TAIL_PROBS.numpy()
    norm = plt.Normalize(min(TRAIN_SIZES) / 1_000.0, max(TRAIN_SIZES) / 1_000.0)
    cmap = plt.colormaps["viridis"]

    line_alpha = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(7.82, 4.4), sharex=True, sharey=True, constrained_layout=True)
    reference_handle = None
    for ax, (model_name, model_rows) in zip(axes, mean_results.groupby("model", sort=False)):
        handle, = ax.plot(
            quantiles,
            reference.to_numpy(dtype=float).clip(min=1e-12),
            color="black",
            linestyle="--",
            lw=1.4,
            alpha=line_alpha,
        )
        if reference_handle is None:
            reference_handle = handle
        for index, (_, row) in enumerate(model_rows.sort_values("n_train").iterrows()):
            n_train = int(row["n_train"])
            y = row[TCE_COLUMNS].to_numpy(dtype=float)
            color = cmap(norm(n_train / 1_000.0))
            ax.plot(
                quantiles,
                y.clip(min=1e-12),
                color=color,
                lw=1.5,
                alpha=line_alpha,
            )

        ax.set_xscale("logit")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)
        ax.set_xlim(float(quantiles[0]), float(quantiles[-1]))
        ax.set_xticks([0.90, 0.99, 0.999, 0.9999])
        ax.set_xticklabels(["90", "99", "99.9", "99.99"])
        ax.grid(True, which="both", alpha=0.4)

    axes[0].set_ylabel("TCE, upper tail log error", fontsize=12)
    for ax in axes:
        ax.set_xlabel("tail quantile (%)", fontsize=12)
    if reference_handle is not None:
        fig.legend([reference_handle], ["test vs true sample"], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=8)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.08,
        pad=0.12,
        aspect=40,
    )
    cbar.set_label("training samples (k)")

    path = Path(__file__).resolve().parent / "_figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


########################################################################################################################
# Main

device, dtype = setup()

dim = DIM
alpha, x_train_full, x_test = load_alpha_stable(device, dtype, dim=dim, n_train=max(TRAIN_SIZES), n_test=N_TEST)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
train_kwargs = make_train_kwargs(device)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s train_sizes=%s",
             device, dtype, dim, n_trials, n_tail, n_mmd, TRAIN_SIZES)

rows = []
for trial in range(1, n_trials + 1):

    logging.info("trial %s/%s", trial, n_trials)

    for n_train in TRAIN_SIZES:
        x_train = x_train_full[:n_train]
        logging.info("n_train=%s", n_train)

        models = {
            "DDPM": DDPMV(net=make_net(dim=dim, device=device, dtype=dtype), dim=dim, n_steps=N_STEPS, device=device),
            "DLPM": DLPMEps(
                net=make_net(dim=dim, device=device, dtype=dtype),
                alpha=alpha,
                dim=dim,
                n_steps=N_STEPS,
                n_trial_A=1,
                n_trial_G=1,
                reduce_type="mean",
                device=device,
            ),
        }

        for model_idx, (name, model) in enumerate(models.items()):
            torch.manual_seed(10_000 * trial + 100 * model_idx + n_train)
            logging.info("train/evaluate %s", name)
            model, _ = train(model, x_train, **train_kwargs)
            rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd, n_train=n_train))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha, n_train=0)


########################################################################################################################
# Plotting

figure_name = f"{Path(__file__).stem}.pdf"
save_training_size_figure(rows, figure_name)
logging.info("saved figure %s", figure_name)
