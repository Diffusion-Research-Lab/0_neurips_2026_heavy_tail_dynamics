"""Visualize 1D sampling paths for DDPMV and GaussianFlowLinear."""

import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from genkit import DDPMV, GaussianFlowLinear, DLPMEps
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import fid, wasserstein_distance
from genkit.nn import MLPModel
from genkit.training import train
from genkit.utils import format_duration
from labkit.report import PRETTY_RCPARAMS
from _utils import print_done, print_model_step, print_start


plt.rcParams.update(PRETTY_RCPARAMS)
t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--blank", action="store_false", help="CI helper.")
args = parser.parse_args()

dim = 1
alpha = 1.5
n_samples = 20_000 if args.blank else 100
n_steps = 128
batch_size = 1024
n_epochs = 128
lr = 1e-3
width = 32
depth = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
fdtype = torch.float32
idtype = torch.int32

x_train, x_val, x_test = fetch_synthetic_data(
    "alpha_stable",
    alpha=alpha,
    n_samples=n_samples,
    dim=dim,
    device=device,
    dtype=fdtype,
)
oracle_w1 = wasserstein_distance(x_test, x_val, p=1)
oracle_fid = fid(x_test, x_val)

print_start(
    "1D path visualization",
    device=device,
    n_samples=n_samples,
    n_steps=n_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    oracle_W1=f"{oracle_w1:.2e}",
    oracle_FID=f"{oracle_fid:.2e}",
)

train_kwargs = dict(target_data=x_train, batch_size=batch_size, n_epochs=n_epochs,
                    num_workers=0, lr=lr, device=device)
net_kwargs = dict(width=width, depth=depth)
model_specs = [("DDPM", DDPMV, "tab:orange", {}),
               ("GF-Linear", GaussianFlowLinear, "tab:blue", {}),
               ("DLPM", DLPMEps, "tab:blue", {'alpha': alpha, "reduce_type": "median"}),
               ]

trained_generators = {}
for title, model_cls, color, extra_kwargs in model_specs:
    print_model_step("TRAIN", title, epochs=n_epochs, batch_size=batch_size, lr=f"{lr:.1e}")
    net = MLPModel(dim=dim, **net_kwargs).to(device=device, dtype=fdtype)
    generator = model_cls(
        net=net,
        dim=dim,
        n_steps=n_steps,
        device=device,
        fdtype=fdtype,
        idtype=idtype,
        **extra_kwargs,
    )
    generator, _ = train(generative_model=generator, **train_kwargs)
    trained_generators[title] = {"generator": generator, "color": color}

figures_dir = Path("_figures")
figures_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(4.5, 3.5 * len(model_specs)), dpi=150)
outer = fig.add_gridspec(len(model_specs), 1, hspace=0.45)

for row, (title, _, _, _) in enumerate(model_specs):
    model_plot = trained_generators[title]
    generator = model_plot["generator"]
    color = model_plot["color"]
    x_gen = generator.sample(n_samples=x_test.size(0))
    w1_value = wasserstein_distance(x_test, x_gen, p=1)
    fid_value = fid(x_test, x_gen)
    print_model_step(
        "EVAL",
        title,
        W1=f"{w1_value:.2e}",
        FID=f"{fid_value:.2e}",
        oracle_W1=f"{oracle_w1:.2e}",
        oracle_FID=f"{oracle_fid:.2e}",
    )

    _, trajectories = generator._sample(1000)
    x = np.stack([state.detach().cpu().numpy().reshape(-1) for state in trajectories], axis=0).T

    _, n_local_steps = x.shape
    timesteps = np.arange(1, n_local_steps + 1)

    inner = outer[row].subgridspec(1, 3, width_ratios=[1, 2.25, 1], wspace=0.0)
    ax_left = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[0, 1], sharey=ax_left)
    ax_right = fig.add_subplot(inner[0, 2], sharey=ax_left)

    for path in x:
        ax_main.plot(timesteps, path, lw=0.5, alpha=0.1, color=color)

    ax_main.set_xlim(1, n_local_steps)
    ax_main.set_xlabel("T", fontsize=7)
    ax_main.set_xticks([1, n_local_steps], ["t=1", f"T={n_local_steps - 1}"], fontsize=7)
    ax_main.set_yticks([])
    ax_main.set_ylim(-7.0, 17.0)
    ax_main.spines["left"].set_visible(False)
    ax_main.tick_params(axis="y", length=0)
    ax_main.set_title(
        f"{title} | W1={w1_value:.2e} ({oracle_w1:.2e}) | FID={fid_value:.2e} ({oracle_fid:.2e})",
        fontsize=8,
    )

    for values, hist_ax, invert in [(x[:, 0], ax_left, True), (x[:, -1], ax_right, False)]:
        hist_ax.hist(values, bins=50, orientation="horizontal", color=color, alpha=0.4)
        if invert:
            hist_ax.invert_xaxis()
        hist_ax.set_xticks([])
        hist_ax.spines["left"].set_visible(False)
        hist_ax.spines["bottom"].set_visible(False)
        hist_ax.tick_params(axis="both", left=False, bottom=False, labelleft=False, labelbottom=False)

fig.tight_layout()
filepath = figures_dir / "visu_1d_path.pdf"
fig.savefig(filepath)
print_done(saved=filepath, total=format_duration(time.time() - t0))
