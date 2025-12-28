"""Simple generation illustrative example."""

# Authors: Hamza Cherkaoui

import time
import torch
from cauda.model import LightNet
from cauda.utils import set_seed, get_device
from cauda.datasets import fetch_synthetic_data
from cauda.training import train
from cauda.metrics import fid, msle_at_quantile
from cauda.plotting import plot_training_loss, plot_scatter
from cauda.diffusion import DDPMEps, DDPMX0
from cauda.flow import GaussianFlowLinear, GaussianFlowOT, GaussianFlowDDPM


####################################################################################################
# Settings
t0_global = time.perf_counter()

dim = 2
n_samples = 5_000
n_steps = 100
batch_size = 512
n_epochs = 100
lr = 5e-3
seed = 4620518
target_data_type = 'spiral'

set_seed(seed)
device = get_device()
dtype = torch.float64
torch.set_default_dtype(dtype)

def msle(x, x_ref):
    return msle_at_quantile(x.abs(), x_ref.abs())

l_metrics = [msle, fid]
l_coefs = [10.0, 1e-3]

####################################################################################################
# Main
for gen_cls in [DDPMEps, DDPMX0, GaussianFlowLinear, GaussianFlowDDPM, GaussianFlowOT]:

    print(f"[INFO] Running experiment on '{target_data_type}' data with '{gen_cls.__name__}' model: ")

    X_train, X_val, X_test = fetch_synthetic_data(target_data_type, n_samples=n_samples,
                                                  dim=dim, device=device, dtype=dtype)

    net = LightNet(dim=dim).to(device=device, dtype=dtype)
    generator = gen_cls(net=net, dim=dim, dtype=dtype, device=device, n_steps=n_steps)
    _, meta = train(generator, target_data=X_train, batch_size=batch_size, n_epochs=n_epochs,
                    lr=lr, device=device)
    X_test_gen = generator.sample(n_samples=n_samples)

    print("[INFO] Evaluation:")
    for coef, metric_func in zip(l_coefs, l_metrics):
        score = metric_func(X_test, X_test_gen)
        ref_score = metric_func(X_test, X_val)
        print(f"       {metric_func.__name__} = {score:.4f} (baseline at {ref_score:.2e})")

####################################################################################################
# Plotting
    filename = plot_training_loss(l_loss=meta['training_loss'], xlogscale=True, ylogscale=True,
                                  plot_dir='_figures', suffix=f"_{target_data_type}_{gen_cls.__name__}")
    print(f"[INFO] Saving '{filename}'")

    if dim == 2:
        filename = plot_scatter(X_test_gen, X_test, plot_dir='_figures',
                                suffix=f"_{target_data_type}_{gen_cls.__name__}")
        print(f"[INFO] Saving '{filename}'")

####################################################################################################
# Timing
print(f"[INFO] Experiment runtime: {time.perf_counter() - t0_global:.1f} s.")
