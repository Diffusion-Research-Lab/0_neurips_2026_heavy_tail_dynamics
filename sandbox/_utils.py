import os
import pandas as pd
import torch
from genkit._noise import sample_scaled_isotropic_alpha_stable
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import mmd_rbf, tail_coverage_error
from genkit.nn import MLPModel
from sandbox._constants import (
    ALPHA,
    DEPTH,
    DEVICE,
    DIM,
    DTYPE,
    MARKERS,
    N_MMD,
    N_TEST,
    N_TRAIN,
    N_TRIALS,
    N_VAL,
    REFERENCE_MODELS,
    SEED,
    TCE_COLUMNS,
    TCE_QUANTILES,
    TCE_TAIL_PROBS,
    WIDTH,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp/flowbench-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402


def load_alpha_stable_data():
    n_tot = N_TRAIN + N_VAL + N_TEST
    torch.manual_seed(SEED)
    x_train, x_val, x_test = fetch_synthetic_data(
        "alpha_stable",
        alpha=ALPHA,
        dim=DIM,
        n_samples=n_tot,
        val_size=N_VAL / n_tot,
        test_size=N_TEST / n_tot,
        standardize=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    if x_train.shape[0] < N_TRAIN:
        x_train = torch.cat([x_train, x_val[: N_TRAIN - x_train.shape[0]]], dim=0)
    else:
        x_train = x_train[:N_TRAIN]
    return x_train, x_test[:N_TEST]


def make_net(dim=DIM, width=WIDTH, depth=DEPTH, input_dim=None, output_dim=None):
    input_dim = dim if input_dim is None else input_dim
    output_dim = dim if output_dim is None else output_dim
    return MLPModel(input_dim=input_dim, output_dim=output_dim, width=width, depth=depth).to(device=DEVICE, dtype=DTYPE)


@torch.no_grad()
def evaluate_samples(name, trial, x_ref, x_gen):
    x_ref = x_ref.detach().cpu().to(torch.float64)
    x_gen = x_gen.detach().cpu().to(torch.float64)

    row = {
        "trial": trial,
        "model": name,
        "mmd_rbf": mmd_rbf(x_ref[:N_MMD], x_gen[:N_MMD]),
    }
    for label, prob in zip(TCE_COLUMNS, TCE_TAIL_PROBS):
        row[label] = tail_coverage_error(
            x_ref,
            x_gen,
            probs=prob.reshape(1),
            tail="upper",
            mode="log",
        )
    return row


@torch.no_grad()
def add_test_vs_true_sample(rows, x_test):
    for trial in range(1, N_TRIALS + 1):
        torch.manual_seed(10_000 + trial)
        x_true = sample_scaled_isotropic_alpha_stable(
            n_samples=N_TEST,
            dim=DIM,
            alpha=ALPHA,
            device=torch.device("cpu"),
            dtype=DTYPE,
        )
        rows.append(evaluate_samples("test vs true sample", trial, x_test, x_true))


def plot_tail_results(rows, title=None, show_std=True):
    results = pd.DataFrame(rows)
    grouped = results.groupby("model")
    mean_results = grouped[["mmd_rbf", *TCE_COLUMNS]].mean()
    std_results = grouped[TCE_COLUMNS].std().fillna(0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (model_name, row) in enumerate(mean_results.iterrows()):
        x = TCE_QUANTILES.numpy()
        y = row[TCE_COLUMNS].to_numpy(dtype=float)
        yerr = std_results.loc[model_name, TCE_COLUMNS].to_numpy(dtype=float)
        y_plot = y.clip(min=1e-12)
        color = "black" if model_name in REFERENCE_MODELS else None
        line, = ax.plot(
            x,
            y_plot,
            marker=MARKERS[i % len(MARKERS)],
            markevery=4,
            label=f"{model_name} (MMD-RBF={row['mmd_rbf']:.1e})",
            lw=2.0,
            alpha=0.7,
            color=color,
        )
        if show_std:
            ax.fill_between(x, (y - yerr).clip(min=1e-12), y + yerr, color=line.get_color(), alpha=0.2)

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xticks([0.5, 0.90, 0.99, 0.999, 0.9999])
    ax.set_xticklabels(["50", "90", "99", "99.9", "99.99"])
    ax.set_xlim(float(TCE_QUANTILES[0]), float(TCE_QUANTILES[-1]))
    ax.set_xlabel("tail quantile (%)", fontsize=12)
    ax.set_ylabel("TCE, upper tail log error", fontsize=12)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax
