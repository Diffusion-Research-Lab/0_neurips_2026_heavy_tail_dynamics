"""Simple generation illustrative example."""

import argparse
import time
import torch
from cauda.model import LightNet
from cauda.datasets import fetch_synthetic_data
from cauda.training import train
from cauda.metrics import mse, msle_90
from cauda.plotting import plot_training_loss, plot_scatter
from cauda.diffusion import DDPMEps, DDPMX0
from cauda.flow import GaussianFlowLinear, GaussianFlowOT, GaussianFlowDDPM


####################################################################################################
# Settings
t0_global = time.perf_counter()

parser = argparse.ArgumentParser(description="Run spiral generation example.")
parser.add_argument("--smoke", action="store_true", help="Run a fast smoke-test configuration.")
parser.add_argument("--no-plot", action="store_true", help="Disable figure generation.")
args = parser.parse_args()

if args.smoke:
    n_samples = 512
    n_steps = 20
    batch_size = 128
    n_epochs = 3
    lr = 5e-3
    l_generators = [DDPMEps, GaussianFlowLinear]
    num_workers = 0
else:
    n_samples = 5_000
    n_steps = 100
    batch_size = 512
    n_epochs = 512
    lr = 1e-3
    l_generators = [DDPMEps, DDPMX0, GaussianFlowLinear, GaussianFlowDDPM, GaussianFlowOT]
    num_workers = 2

dim = 2
target_data_type = "spiral"
device = 'cpu'
dtype = torch.float64


l_metrics = [msle_90, mse]
l_coefs = [10.0, 1.0]

####################################################################################################
# Main
for gen_cls in l_generators:

    print(f"[INFO] Running experiment on '{target_data_type}' data with '{gen_cls.__name__}' model: ")

    X_train, X_val, X_test = fetch_synthetic_data(target_data_type, n_samples=n_samples,
                                                  dim=dim, device=device, dtype=dtype)

    net = LightNet(dim=dim, width=64, n_blocks=4).to(device=device, dtype=dtype)
    generator = gen_cls(net=net, dim=dim, dtype=dtype, device=device, n_steps=n_steps)
    _, diagnostics = train(generator, target_data=X_train, batch_size=batch_size, n_epochs=n_epochs,
                           lr=lr, device=device, num_workers=num_workers)
    X_test_gen = generator.sample(n_samples=n_samples)

    print("[INFO] Evaluation:")
    for coef, metric_func in zip(l_coefs, l_metrics):
        score = metric_func(X_test, X_test_gen)
        ref_score = metric_func(X_test, X_val)
        print(f"       {metric_func.__name__} = {score:.2e} (baseline at {ref_score:.2e})")

####################################################################################################
# Plotting
    if not args.no_plot and not args.smoke:
        filename = plot_training_loss(l_loss=diagnostics['visitors']['core']['training_loss'],
                                      xlogscale=True, ylogscale=True,
                                      plot_dir='_figures', suffix=f"_{target_data_type}_{gen_cls.__name__}")
        print(f"[INFO] Saving '{filename}'")

        if dim == 2:
            filename = plot_scatter(X_test_gen, X_test, plot_dir='_figures',
                                    suffix=f"_{target_data_type}_{gen_cls.__name__}")
            print(f"[INFO] Saving '{filename}'")

####################################################################################################
# Timing
print(f"[INFO] Experiment runtime: {time.perf_counter() - t0_global:.1f} s.")
