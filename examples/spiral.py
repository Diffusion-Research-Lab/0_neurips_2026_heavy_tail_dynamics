"""Simple 2D generation illustrative example."""

import time
import argparse
import torch
from genkit.model import LightNet
from genkit.datasets import fetch_synthetic_data
from genkit.training import train
from genkit.metrics import mse
from genkit.plotting import plot_training_loss, plot_scatter
from genkit.diffusion import DDPMEps, DDPMX0
from genkit.flow import GaussianFlowLinear, GaussianFlowOT, GaussianFlowDDPM


####################################################################################################
# Settings
t0_global = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument("--blank", action="store_false")
args = parser.parse_args()

n_samples = 10_000 if args.blank else 100
n_steps = 100
batch_size = 256
n_epochs = 512
lr = 1e-3
l_generators = [DDPMEps, DDPMX0, GaussianFlowLinear, GaussianFlowDDPM, GaussianFlowOT]
target_data_type = "spiral"
device = 'cpu'
fdtype = torch.float32
idtype = torch.int32

X_train, X_val, X_test = fetch_synthetic_data(target_data_type, n_samples=n_samples,
                                              device=device, dtype=fdtype)

####################################################################################################
# Main
for gen_cls in l_generators:

    print(f"[INFO] Running experiment on '{target_data_type}' data with '{gen_cls.__name__}' model: ")

    net = LightNet(dim=2, width=128, n_blocks=4).to(device=device, dtype=fdtype)
    generator = gen_cls(net=net, dim=2, fdtype=fdtype, idtype=idtype, device=device, n_steps=n_steps)
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

    filename = plot_scatter(X_test_gen, X_test, plot_dir='_figures',
                            suffix=f"_{target_data_type}_{gen_cls.__name__}")
    print(f"[INFO] Saving '{filename}'")

####################################################################################################
# Timing
print(f"[INFO] Experiment runtime: {time.perf_counter() - t0_global:.1f} s.")
