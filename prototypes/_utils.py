import os
from pathlib import Path
import pandas as pd
import torch
from genkit._noise import sample_scaled_isotropic_alpha_stable
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import mmd_rbf, tail_coverage_error
from genkit.nn import MLPModel
from _constants import (
    ALPHA,
    DEPTH,
    DEVICE,
    DIM,
    DTYPE,
    MARKERS,
    N_TEST,
    N_TRAIN,
    N_VAL,
    REFERENCE_MODELS,
    SAMPLE_CHUNK_SIZE,
    SEED,
    TCE_COLUMNS,
    TCE_TAIL_PROBS,
    TRAIN_KWARGS,
    WIDTH,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flowbench-matplotlib")

import matplotlib  # noqa
matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt  # noqa


def setup(seed=SEED):
    torch.manual_seed(seed)
    return DEVICE, DTYPE


def load_alpha_stable(device, dtype, dim=DIM, alpha=ALPHA, n_train=N_TRAIN, n_test=N_TEST, n_val=N_VAL):
    n_samples = int(n_train) + int(n_val) + int(n_test)
    val_size = int(n_val) / n_samples
    test_size = int(n_test) / n_samples
    x_train, x_val, x_test = fetch_synthetic_data("alpha_stable", alpha=alpha, dim=dim, n_samples=n_samples,
                                                  val_size=val_size, test_size=test_size, standardize=False,
                                                  device=device, dtype=dtype)
    if x_train.shape[0] < n_train:
        x_train = torch.cat([x_train, x_val[: n_train - x_train.shape[0]]], dim=0)
    else:
        x_train = x_train[:n_train]
    x_test = x_test[:n_test]
    if x_train.shape[0] != n_train or x_test.shape[0] != n_test:
        raise RuntimeError(f"unexpected alpha-stable split sizes: train={x_train.shape[0]} test={x_test.shape[0]}")
    return alpha, x_train, x_test


def make_net(dim, device, width=WIDTH, depth=DEPTH, dtype=DTYPE, input_dim=None, output_dim=None):
    input_dim = dim if input_dim is None else input_dim
    output_dim = dim if output_dim is None else output_dim
    return MLPModel(input_dim=input_dim, output_dim=output_dim, width=width, depth=depth).to(device=device, dtype=dtype)


def make_train_kwargs(device=DEVICE):
    kwargs = dict(TRAIN_KWARGS)
    kwargs["device"] = device
    return kwargs


@torch.no_grad()
def _evaluate_samples(name, trial, x_ref, x_gen, n_tail, n_mmd):
    x_ref = x_ref[:n_tail].detach().cpu().to(torch.float64)
    x_gen = x_gen[:n_tail].detach().cpu().to(torch.float64)

    row = {"trial": trial, "model": name, "mmd_rbf": mmd_rbf(x_ref[:n_mmd], x_gen[:n_mmd])}
    for label, prob in zip(TCE_COLUMNS, TCE_TAIL_PROBS):
        row[label] = tail_coverage_error(x_ref, x_gen, probs=prob.reshape(1), tail="upper", mode="log")

    return row


@torch.no_grad()
def evaluate_model(name, model, trial, x_ref, n_tail, n_mmd, chunk_size=SAMPLE_CHUNK_SIZE):
    n_tail = int(n_tail)
    chunk_size = int(chunk_size)
    chunks = []
    for start in range(0, n_tail, chunk_size):
        chunks.append(model.sample(min(chunk_size, n_tail - start)).detach().cpu())
    x_gen = torch.cat(chunks, dim=0)
    return _evaluate_samples(name, trial, x_ref, x_gen, n_tail, n_mmd)


def add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=1.7):
    x_ref = x_test.detach().cpu().to(torch.float64)
    dim = x_test.reshape(x_test.shape[0], -1).shape[1]
    for trial in range(1, n_trials + 1):
        torch.manual_seed(10_000 + trial)
        x_true = sample_scaled_isotropic_alpha_stable(n_samples=len(x_test), dim=dim, alpha=alpha, device=torch.device("cpu"), dtype=x_test.dtype)
        rows.append(_evaluate_samples("test vs true sample", trial, x_ref, x_true, n_tail, n_mmd))


def save_tail_figure(rows, title, filename):
    results = pd.DataFrame(rows)
    tce_cols = TCE_COLUMNS
    grouped = results.groupby("model")
    mean_results = grouped[["mmd_rbf", *tce_cols]].mean()
    std_results = grouped[tce_cols].std().fillna(0.0)
    x = 1.0 - TCE_TAIL_PROBS.numpy()

    line_alpha = 0.5

    fig, ax = plt.subplots(figsize=(4.8, 6.0))
    for index, (model_name, row) in enumerate(mean_results.iterrows()):

        y = row[tce_cols].to_numpy(dtype=float)
        yerr = std_results.loc[model_name, tce_cols].to_numpy(dtype=float)
        color = "black" if model_name in REFERENCE_MODELS else None

        line, = ax.plot(x, y, marker=MARKERS[index % len(MARKERS)], markevery=4, linewidth=1.8,
                        label=f"{model_name} (MMD-RBF={row['mmd_rbf']:.1e})", alpha=line_alpha, color=color)
        ax.fill_between(x, (y - yerr).clip(min=1e-12), y + yerr, color=line.get_color(), alpha=0.2)

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xticks([0.90, 0.99, 0.999, 0.9999, 0.99999])
    ax.set_xticklabels(["90", "99", "99.9", "99.99", "99.999"])
    ax.set_xlabel("tail quantile (%)", fontsize=12)
    ax.set_ylabel("TCE, upper tail log error", fontsize=12)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=8)

    fig.tight_layout()

    path = Path(__file__).resolve().parent / "_figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
