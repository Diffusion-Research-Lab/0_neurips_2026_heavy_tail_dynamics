import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flowbench-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import mmd_rbf, tail_coverage_error
from genkit.nn import MLPModel


TCE_TAIL_PROBS = torch.logspace(-1.0, -5.0, 20, dtype=torch.float64)
DEFAULT_MMD_SAMPLES = 10_000
DEFAULT_SAMPLE_CHUNK_SIZE = 10_000


def _tce_label(prob):
    quantile = 100.0 * (1.0 - float(prob))
    precision = 2 if quantile < 99.0 else 4
    label = f"{quantile:.{precision}f}".rstrip("0").rstrip(".")
    return f"TCE({label})"


TCE_PROBS = {_tce_label(prob): float(prob) for prob in TCE_TAIL_PROBS}


def setup(seed=0):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"[INFO] device={device} dtype={dtype}")
    return device, dtype


def load_alpha_stable(device, dtype, alpha=1.7, n_train=30_000, n_test=1_500_000, n_val=1):
    n_samples = int(n_train) + int(n_val) + int(n_test)
    val_size = int(n_val) / n_samples
    test_size = int(n_test) / n_samples
    print(
        f"[INFO] loading isotropic alpha-stable data alpha={alpha} "
        f"target_train={n_train} target_test={n_test}"
    )
    x_train, x_val, x_test = fetch_synthetic_data(
        "alpha_stable",
        alpha=alpha,
        dim=2,
        n_samples=n_samples,
        val_size=val_size,
        test_size=test_size,
        standardize=False,
        device=device,
        dtype=dtype,
    )
    if x_train.shape[0] < n_train:
        x_train = torch.cat([x_train, x_val[: n_train - x_train.shape[0]]], dim=0)
    else:
        x_train = x_train[:n_train]
    x_test = x_test[:n_test]
    if x_train.shape[0] != n_train or x_test.shape[0] != n_test:
        raise RuntimeError(f"unexpected alpha-stable split sizes: train={x_train.shape[0]} test={x_test.shape[0]}")
    print(f"[INFO] train_shape={x_train.shape[0]} test_shape={x_test.shape[0]}")
    return alpha, x_train, x_test


def fetch_alpha_stable_reference(n_samples, device, dtype, alpha=1.7, seed=0):
    n_samples = int(n_samples)
    n_total = n_samples + 2
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        _, _, x_ref = fetch_synthetic_data(
            "alpha_stable",
            alpha=alpha,
            dim=2,
            n_samples=n_total,
            val_size=1.0 / n_total,
            test_size=n_samples / n_total,
            standardize=False,
            device=device,
            dtype=dtype,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    return x_ref


def make_mlp(device, dtype):
    return MLPModel(dim=2, width=128, depth=3).to(device=device, dtype=dtype)


def make_train_kwargs(device):
    return dict(
        batch_size=128,
        n_epochs=128,
        lr=5e-4,
        device=device,
        use_adamw=False,
        lr_schedule="constant",
        freq_logging=8,
    )


def evaluation_sizes(x_test):
    n_tail = x_test.shape[0]
    return n_tail, min(DEFAULT_MMD_SAMPLES, n_tail)


@torch.no_grad()
def sample_model_in_chunks(model, n_samples, chunk_size=DEFAULT_SAMPLE_CHUNK_SIZE):
    n_samples = int(n_samples)
    if chunk_size is None or int(chunk_size) >= n_samples:
        return model.sample(n_samples)
    chunks = []
    for start in range(0, n_samples, int(chunk_size)):
        chunks.append(model.sample(min(int(chunk_size), n_samples - start)).detach().cpu())
    return torch.cat(chunks, dim=0)


def figure_path(filename):
    path = Path(__file__).resolve().parent / "_figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def evaluate_samples(name, trial, x_ref, x_gen, n_tail, n_mmd):
    x_ref = x_ref[:n_tail].detach().cpu().to(torch.float64)
    x_gen = x_gen[:n_tail].detach().cpu().to(torch.float64)
    row = {
        "trial": trial,
        "model": name,
        "mmd_rbf": mmd_rbf(x_ref[:n_mmd], x_gen[:n_mmd]),
    }
    for label, prob in TCE_PROBS.items():
        row[label] = tail_coverage_error(
            x_ref,
            x_gen,
            probs=torch.tensor([prob], dtype=torch.float64),
            tail="upper",
            mode="log",
        )
    return row


@torch.no_grad()
def evaluate_model(name, model, trial, x_ref, n_tail, n_mmd):
    print(f"[INFO] evaluating trial={trial} model={name} n_tail={n_tail} n_mmd={n_mmd}")
    x_gen = sample_model_in_chunks(model, n_tail)
    return evaluate_samples(name, trial, x_ref, x_gen, n_tail, n_mmd)


@torch.no_grad()
def evaluate_model_with_source(name, model, trial, x_ref, sample_source, n_tail, n_mmd, chunk_size=None):
    print(f"[INFO] evaluating trial={trial} model={name} n_tail={n_tail} n_mmd={n_mmd}")
    sample_source = sample_source[:n_tail]
    if chunk_size is None:
        x_gen = model.sample(sample_source=sample_source)
    else:
        x_gen = model.sample(sample_source=sample_source, chunk_size=chunk_size)
    return evaluate_samples(name, trial, x_ref, x_gen, n_tail, n_mmd)


def add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=1.7):
    print("[INFO] computing test-vs-true-sample reference metrics")
    x_ref = x_test.detach().cpu().to(torch.float64)
    reference_tail = min(n_tail, x_ref.shape[0])
    reference_mmd = min(n_mmd, reference_tail)
    print(f"[INFO] test-vs-true-sample n_tail={reference_tail} n_mmd={reference_mmd}")

    for trial in range(1, n_trials + 1):
        x_true = fetch_alpha_stable_reference(
            reference_tail,
            device=torch.device("cpu"),
            dtype=x_test.dtype,
            alpha=alpha,
            seed=10_000 + trial,
        )
        rows.append(evaluate_samples("test vs true sample", trial, x_ref, x_true, reference_tail, reference_mmd))


def save_tail_figure(rows, title, filename):
    print("[INFO] aggregating results")
    results = pd.DataFrame(rows)
    tce_cols = list(TCE_PROBS)
    grouped = results.groupby("model")
    mean_results = grouped[["mmd_rbf", *tce_cols]].mean()
    std_results = grouped[tce_cols].std().fillna(0.0)
    tce_x = 1.0 - TCE_TAIL_PROBS.numpy()
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
    reference_models = {"test vs test", "test vs true sample"}
    line_alpha = 0.5

    fig, ax = plt.subplots(figsize=(6, 5))
    for index, (model_name, row) in enumerate(mean_results.iterrows()):
        y = row[tce_cols].to_numpy(dtype=float)
        yerr = std_results.loc[model_name, tce_cols].to_numpy(dtype=float)
        color = "black" if model_name in reference_models else None
        line, = ax.plot(
            tce_x,
            y,
            marker=markers[index % len(markers)],
            markevery=4,
            linewidth=1.8,
            label=f"{model_name} (MMD-RBF={row['mmd_rbf']:.1e})",
            alpha=line_alpha,
            color=color,
        )
        ax.fill_between(tce_x, (y - yerr).clip(min=1e-12), y + yerr, color=line.get_color(), alpha=0.2)

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xticks([0.90, 0.99, 0.999, 0.9999, 0.99999])
    ax.set_xticklabels(["90", "99", "99.9", "99.99", "99.999"])
    ax.set_xlabel("tail quantile (%)", fontsize=12)
    ax.set_ylabel("TCE, upper tail log error", fontsize=12)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    path = figure_path(filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote figure to {path}")
