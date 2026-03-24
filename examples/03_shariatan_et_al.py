"""Compare DLPM variants on 2D alpha-stable data."""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from genkit import DDPMV, DLPMEpsOrigin
from genkit.datasets import fetch_synthetic_data
from genkit.nn import MLPModel
from genkit.plotting import PRETTY_RCPARAMS
from genkit.training import train
from genkit.utils import format_duration
from genkit.visitor import CoreMetricsVisitor
from _utils import plot_generated_samples


plt.rcParams.update(PRETTY_RCPARAMS)

t0_global = time.perf_counter()
figures_dir = Path("_figures")
figures_dir.mkdir(parents=True, exist_ok=True)
dim = 2
alpha_data = 1.8
n_train_samples = 30_000
n_steps = 100
batch_size = 1024
n_epochs = 20
lr = 5e-3
n_trials = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
fdtype = torch.float32
idtype = torch.int32

model_specs = [
    {"name": "DDPM", "cls": DDPMV, "gen_kwargs": {}},
    {"name": "DLPM(1.7)", "cls": DLPMEpsOrigin, "gen_kwargs": {"alpha": 1.7, "monte_carlo_outer": 1}},
    {"name": "DLPM(1.8)", "cls": DLPMEpsOrigin, "gen_kwargs": {"alpha": 1.8, "monte_carlo_outer": 1}},
    {"name": "DLPM(1.9)", "cls": DLPMEpsOrigin, "gen_kwargs": {"alpha": 1.9, "monte_carlo_outer": 1}},
]

data_kwargs = dict(alpha=alpha_data, n_samples=n_train_samples, dim=dim, device=device, dtype=fdtype)
x_train, _, _ = fetch_synthetic_data(target_data="alpha_stable", **data_kwargs)

train_kwargs = dict(target_data=x_train, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                    device=device, visitors=[CoreMetricsVisitor()])
net_kwargs = dict(width=64, depth=4, time_dim=32, dropout=0.0, use_norm=True)

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

        t0 = time.perf_counter()
        print(f"[{run_idx:02d}/{total_runs:02d}] {spec['name']}: training...", end="")
        generator, diagnostic = train(generative_model=generator, **train_kwargs)
        print(f" done ({format_duration(time.perf_counter() - t0)}).")

        generators.append(generator)
        diagnostics.append(diagnostic)

    results.append({"generators": generators, "diagnostics": diagnostics})

plot_generated_samples(results, model_specs, data_kwargs, n_trials, figures_dir / "shariatan_et_al_samples.pdf")

print(f"[INFO] Saved figures in {figures_dir}")
print(f"[INFO] Total runtime: {format_duration(time.perf_counter() - t0_global)}.")
