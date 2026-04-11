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
from labkit.report import PRETTY_RCPARAMS
from _utils import plot_generated_samples, print_done, print_model_step, print_start


plt.rcParams.update(PRETTY_RCPARAMS)

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

print_start(
    "Shariatan et al comparison",
    device=device,
    alpha=alpha_data,
    n_samples=n_train_samples,
    n_steps=n_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    n_trials=n_trials,
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
        print_model_step(
            "TRAIN",
            spec["name"],
            index=f"{run_idx:02d}/{total_runs:02d}",
            epochs=n_epochs,
            batch_size=batch_size,
            lr=f"{lr:.1e}",
        )
        generator, diagnostic = train(generative_model=generator, **train_kwargs)
        print_model_step("EVAL", spec["name"], index=f"{run_idx:02d}/{total_runs:02d}", elapsed=f"{time.time() - t0:.1f}s")

        generators.append(generator)
        diagnostics.append(diagnostic)

    results.append({"generators": generators, "diagnostics": diagnostics})

plot_generated_samples(results, model_specs, data_kwargs, n_trials, figures_dir / "shariatan_et_al_samples.pdf")

print_done(saved=figures_dir)
