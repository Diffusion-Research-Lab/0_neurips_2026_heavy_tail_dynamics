"""Visualize 1D sampling paths for DDPMV and GaussianFlowLinear."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from genkit import DDPMV, GaussianFlowLinear
from genkit.datasets import fetch_synthetic_data
from genkit.nn import MLPModel
from genkit.plotting import PRETTY_RCPARAMS
from genkit.training import train
from _utils import plot_path_panel


plt.rcParams.update(PRETTY_RCPARAMS)


parser = argparse.ArgumentParser()
parser.add_argument("--blank", action="store_false", help="CI helper.")
args = parser.parse_args()

dim = 1
n_samples = 10_000 if args.blank else 100
n_steps = 250
batch_size = 1024
n_epochs = 100
lr = 5e-4
width = 64
depth = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
fdtype = torch.float32
idtype = torch.int32

x_train, _, _ = fetch_synthetic_data("balanced_bimodal_gaussian", n_samples=n_samples, dim=dim,
                                     device=device, dtype=fdtype)

train_kwargs = dict(target_data=x_train, batch_size=batch_size, n_epochs=n_epochs,
                    lr=lr, device=device)
net_kwargs = dict(width=width, depth=depth)
model_specs = [("DDPMV", DDPMV, "tab:orange"),
               ("GaussianFlowLinear", GaussianFlowLinear, "tab:blue"),
               ]

trained_generators = {}
for title, model_cls, color in model_specs:
    net = MLPModel(dim=dim, **net_kwargs).to(device=device, dtype=fdtype)
    generator = model_cls(
        net=net,
        dim=dim,
        n_steps=n_steps,
        device=device,
        fdtype=fdtype,
        idtype=idtype,
    )
    generator, _ = train(generative_model=generator, **train_kwargs)
    trained_generators[title] = {"generator": generator, "color": color}

figures_dir = Path("_figures")
figures_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(4.5, 4.0), dpi=150)
outer = fig.add_gridspec(len(model_specs), 1, hspace=0.45)

for row, (title, _, _) in enumerate(model_specs):
    model_plot = trained_generators[title]
    plot_path_panel(model_plot["generator"], fig, outer[row], title, color=model_plot["color"])

fig.tight_layout()
fig.savefig(figures_dir / "visu_1d_path.pdf")
plt.close(fig)
