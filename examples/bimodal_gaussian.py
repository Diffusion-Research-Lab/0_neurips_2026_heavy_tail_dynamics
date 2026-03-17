"""Simple 1D generation illustrative example."""

import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from genkit.model import LightNet
from genkit.datasets import fetch_synthetic_data
from genkit.training import train
from genkit.metrics import mse
from genkit.plotting import plot_training_loss
from genkit.diffusion import DDPMEps, DDPMX0
from genkit.flow import GaussianFlowLinear, GaussianFlowOT, GaussianFlowDDPM
from genkit.plotting import PRETTY_RCPARAMS


plt.rcParams.update(PRETTY_RCPARAMS)

####################################################################################################
# Settings
t0_global = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument("--blank", action="store_false")
args = parser.parse_args()

n_samples = 5_000 if args.blank else 100
n_steps = 100
batch_size = 128
n_epochs = 256
lr = 1e-3
l_generators = [DDPMEps, DDPMX0, GaussianFlowLinear, GaussianFlowDDPM, GaussianFlowOT]
dim = 1
target_data_type = "balanced_bimodal_gaussian"
device = 'cpu'
fdtype = torch.float32
idtype = torch.int32

X_train, X_val, X_test = fetch_synthetic_data(target_data_type, n_samples=n_samples,
                                              dim=dim, device=device, dtype=fdtype)

####################################################################################################
# Main
for gen_cls in l_generators:

    print(f"[INFO] Running experiment on '{target_data_type}' data with '{gen_cls.__name__}' model: ")

    net = LightNet(dim=dim, width=64, n_blocks=2).to(device=device, dtype=fdtype)
    generator = gen_cls(net=net, dim=dim, fdtype=fdtype, idtype=idtype, device=device, n_steps=n_steps)
    _, diagnostics = train(generator, target_data=X_train, batch_size=batch_size, n_epochs=n_epochs,
                           lr=lr, device=device)
    X_test_gen = generator.sample(n_samples=n_samples)

    print(f"[INFO] Evaluation: MSE = {mse(X_test, X_test_gen):.2e} (baseline at {mse(X_test, X_val):.2e})")

####################################################################################################
# Plotting
    filename = plot_training_loss(l_loss=diagnostics['visitors']['core']['training_loss'],
                                  xlogscale=True, ylogscale=True, plot_dir='_figures',
                                  suffix=f"_{target_data_type}_{gen_cls.__name__}")
    print(f"[INFO] Saving '{filename}'")

    alpha = 0.5
    bins = 50
    filename = Path('_figures') / f"_{target_data_type}_{gen_cls.__name__}_histograms.pdf"
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(4.0, 2.0), squeeze=False)
    axis[0, 0].hist(X_test, bins=bins, density=True, alpha=alpha, label='Ref.')
    axis[0, 0].hist(X_test_gen, bins=bins, density=True, alpha=alpha, label='Gen.')
    axis[0, 0].legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()
    print(f"[INFO] Saving '{filename}'")

####################################################################################################
# Timing
print(f"[INFO] Experiment runtime: {time.perf_counter() - t0_global:.1f} s.")
