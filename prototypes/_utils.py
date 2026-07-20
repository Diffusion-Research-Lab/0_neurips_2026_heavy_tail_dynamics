import os
from collections.abc import Callable
from pathlib import Path
import pandas as pd
import torch
from genkit._noise import sample_scaled_isotropic_alpha_stable
from genkit.metrics import mmd_rbf
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
    if int(n_val) < 0:
        raise ValueError("n_val must be non-negative.")
    x_train = sample_scaled_isotropic_alpha_stable(n_samples=int(n_train), dim=int(dim), alpha=alpha, device=device, dtype=dtype)
    x_test = sample_scaled_isotropic_alpha_stable(n_samples=int(n_test), dim=int(dim), alpha=alpha, device=torch.device("cpu"), dtype=dtype)
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
def _metric_matrix(x, n_rows=None):
    if n_rows is not None:
        x = x[:int(n_rows)]
    return x.detach().cpu().reshape(x.shape[0], -1).to(torch.float32)


@torch.no_grad()
def _coverage_counts(x, thresholds, chunk_size=SAMPLE_CHUNK_SIZE):
    counts = torch.zeros(thresholds.shape, dtype=torch.float64)
    for start in range(0, x.shape[0], int(chunk_size)):
        block = x[start: start + int(chunk_size)]
        counts += (block.unsqueeze(0) > thresholds.unsqueeze(1)).sum(dim=1).to(torch.float64)
    return counts


def _tail_errors(ref_counts, gen_counts, n_ref, n_gen, eps=1e-12):
    ref_cov = ref_counts / float(n_ref)
    gen_cov = gen_counts / float(n_gen)
    return (torch.log(gen_cov + float(eps)) - torch.log(ref_cov + float(eps))).abs().mean(dim=1)


@torch.no_grad()
def evaluate_sampler(name, sampler: Callable[[int], torch.Tensor], trial, x_ref, n_tail, n_mmd, chunk_size=SAMPLE_CHUNK_SIZE, **row_values):
    n_tail = int(n_tail)
    n_mmd = min(int(n_mmd), n_tail)
    chunk_size = int(chunk_size)
    if n_tail <= 0:
        raise ValueError("n_tail must be positive.")
    if n_mmd < 2:
        raise ValueError("n_mmd must be at least 2.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    x_ref = _metric_matrix(x_ref, n_tail)
    quantiles = (1.0 - TCE_TAIL_PROBS).to(dtype=x_ref.dtype)
    thresholds = torch.quantile(x_ref, quantiles, dim=0)
    ref_counts = _coverage_counts(x_ref, thresholds, chunk_size=chunk_size)

    gen_counts = torch.zeros_like(ref_counts)
    mmd_chunks = []
    mmd_remaining = n_mmd
    for start in range(0, n_tail, chunk_size):
        size = min(chunk_size, n_tail - start)
        x_gen = _metric_matrix(sampler(size))
        if x_gen.shape[0] != size:
            raise RuntimeError(f"sampler returned {x_gen.shape[0]} samples, expected {size}.")
        gen_counts += _coverage_counts(x_gen, thresholds, chunk_size=chunk_size)
        if mmd_remaining > 0:
            mmd_chunks.append(x_gen[:mmd_remaining])
            mmd_remaining -= min(mmd_remaining, x_gen.shape[0])

    x_ref_mmd = x_ref[:n_mmd]
    x_gen_mmd = torch.cat(mmd_chunks, dim=0)[:n_mmd]
    row = {"trial": trial, "model": name, **row_values, "mmd_rbf": mmd_rbf(x_ref_mmd, x_gen_mmd)}
    for label, value in zip(TCE_COLUMNS, _tail_errors(ref_counts, gen_counts, n_tail, n_tail)):
        row[label] = float(value.item())

    return row


@torch.no_grad()
def _evaluate_samples(name, trial, x_ref, x_gen, n_tail, n_mmd, **row_values):
    x_gen = _metric_matrix(x_gen, n_tail)
    return evaluate_sampler(name, lambda size: x_gen[:size], trial, x_ref, n_tail=x_gen.shape[0], n_mmd=n_mmd, chunk_size=x_gen.shape[0], **row_values)


@torch.no_grad()
def evaluate_model(name, model, trial, x_ref, n_tail, n_mmd, chunk_size=SAMPLE_CHUNK_SIZE, **row_values):
    return evaluate_sampler(name, model.sample, trial, x_ref, n_tail, n_mmd, chunk_size=chunk_size, **row_values)


def add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=1.7, **row_values):
    dim = x_test.reshape(x_test.shape[0], -1).shape[1]
    for trial in range(1, n_trials + 1):
        torch.manual_seed(10_000 + trial)
        sampler = lambda size: sample_scaled_isotropic_alpha_stable(n_samples=size, dim=dim, alpha=alpha, device=torch.device("cpu"), dtype=x_test.dtype)
        rows.append(evaluate_sampler("test vs true sample", sampler, trial, x_test, n_tail, n_mmd, **row_values))


def marker_positions(n_points, n_markers=5):
    if n_points <= n_markers:
        return list(range(n_points))
    return sorted(set(torch.linspace(0, n_points - 1, n_markers).round().to(torch.int64).tolist()))


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

        line, = ax.plot(x, y, marker=MARKERS[index % len(MARKERS)], markevery=marker_positions(len(x)), linewidth=1.8,
                        label=f"{model_name} (MMD-RBF={row['mmd_rbf']:.1e})", alpha=line_alpha, color=color)
        ax.fill_between(x, (y - yerr).clip(min=1e-12), y + yerr, color=line.get_color(), alpha=0.2)

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xticks([0.90, 0.99, 0.999, 0.9999])
    ax.set_xticklabels(["90", "99", "99.9", "99.99"])
    ax.set_xlim(float(x[0]), float(x[-1]))
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
