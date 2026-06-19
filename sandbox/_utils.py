import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import mmd_rbf, tail_coverage_error
from genkit.nn import MLPModel


TCE_PROBS = {
    "TCE(90)": 1e-1,
    "TCE(99)": 1e-2,
    "TCE(99,9)": 1e-3,
    "TCE(99,99)": 1e-4,
}


def setup(seed=0):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"[INFO] device={device} dtype={dtype}")
    return device, dtype


def load_alpha_stable(device, dtype, alpha=1.7):
    print(f"[INFO] loading isotropic alpha-stable data alpha={alpha}")
    x_train, _, x_test = fetch_synthetic_data(
        "alpha_stable",
        alpha=alpha,
        dim=2,
        n_samples=170_000,
        val_size=0.001,
        test_size=0.8,
        standardize=False,
        device=device,
        dtype=dtype,
    )
    print(f"[INFO] train_shape={x_train.shape[0]} test_shape={x_test.shape[0]}")
    return alpha, x_train, x_test


def make_mlp(device, dtype):
    return MLPModel(dim=2, width=128, depth=3).to(device=device, dtype=dtype)


@torch.no_grad()
def evaluate_model(name, model, trial, x_ref, n_tail, n_mmd):
    print(f"[INFO] evaluating trial={trial} model={name} n_tail={n_tail} n_mmd={n_mmd}")
    x_ref = x_ref[:n_tail].detach().cpu().to(torch.float64)
    x_gen = model.sample(len(x_ref)).detach().cpu().to(torch.float64)
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


def add_test_vs_test(rows, x_test, n_tail, n_mmd, n_trials):
    print("[INFO] computing test-vs-test reference metrics")
    x_ref = x_test.detach().cpu().to(torch.float64)
    reference_tail = min(n_tail, x_ref.shape[0] // 2)
    reference_mmd = min(n_mmd, reference_tail)
    print(f"[INFO] test-vs-test n_tail={reference_tail} n_mmd={reference_mmd}")

    for trial in range(1, n_trials + 1):
        generator = torch.Generator(device="cpu").manual_seed(10_000 + trial)
        perm = torch.randperm(x_ref.shape[0], generator=generator)
        x_a = x_ref[perm[:reference_tail]]
        x_b = x_ref[perm[reference_tail: 2 * reference_tail]]
        row = {
            "trial": trial,
            "model": "test vs test",
            "mmd_rbf": mmd_rbf(x_a[:reference_mmd], x_b[:reference_mmd]),
        }
        for label, prob in TCE_PROBS.items():
            row[label] = tail_coverage_error(
                x_a,
                x_b,
                probs=torch.tensor([prob], dtype=torch.float64),
                tail="upper",
                mode="log",
            )
        rows.append(row)


def save_tail_figure(rows, title, filename):
    print("[INFO] aggregating results")
    results = pd.DataFrame(rows)
    tce_cols = list(TCE_PROBS)
    grouped = results.groupby("model")
    mean_results = grouped[["mmd_rbf", *tce_cols]].mean()
    std_results = grouped[tce_cols].std().fillna(0.0)
    tce_x = [0.90, 0.99, 0.999, 0.9999]

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, row in mean_results.iterrows():
        y = row[tce_cols].to_numpy(dtype=float)
        yerr = std_results.loc[model_name, tce_cols].to_numpy(dtype=float)
        line, = ax.plot(
            tce_x,
            y,
            marker="o",
            linewidth=2,
            label=f"{model_name} (MMD-RBF={row['mmd_rbf']:.4g})",
            alpha=0.5,
        )
        ax.fill_between(tce_x, (y - yerr).clip(min=1e-12), y + yerr, color=line.get_color(), alpha=0.2)

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xticks(tce_x)
    ax.set_xticklabels(["90", "99", "99.9", "99.99"])
    ax.set_xlabel("tail quantile (%)", fontsize=12)
    ax.set_ylabel("TCE, upper tail log error", fontsize=12)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    path = Path(__file__).resolve().parent / "_figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote figure to {path}")
