"""Heavy tail illustrative example."""

# Authors: Hamza Cherkaoui

import argparse
import pprint
import time
from cauda.model import LightNet
from cauda.datasets import fetch_synthetic_data
from cauda.training import train
from cauda.metrics import msle_at_quantile
from cauda.plotting import plot_heatmap
from cauda.flow import AlphaStableFlowLinear
from labkit.config import load_config


####################################################################################################
# Settings
t0_global = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="bench_2_config.yaml")
args = parser.parse_args()

cfg = load_config(args.config).set_up()

print('-' * 40)
print("Experimental configuration:")
print("---------------------------")
pprint.pprint(cfg.as_dict())
print('-' * 40)

def msle(x, x_ref):
    return msle_at_quantile(x.abs(), x_ref.abs())

results = dict()
for alpha_data in cfg.l_alpha_data:
    for alpha_generator in cfg.l_alpha_generator:
        results[(float(alpha_data), float(alpha_generator))] = []

####################################################################################################
# Main
for n in range(cfg.n_trials):

    print(f"[INFO] Running trial {n + 1:02d} / {cfg.n_trials:02d}")

    for alpha_data in cfg.l_alpha_data:

        X_train, _, X_test = fetch_synthetic_data(cfg.target_data_type, n_samples=cfg.n_samples,
                                                  dim=cfg.dim, alpha=alpha_data, device=cfg.device,
                                                  dtype=cfg.dtype)

        for alpha_generator in cfg.l_alpha_generator:

            t0 = time.perf_counter()

            print(f"[INFO] Running experiment on alpha-data={alpha_data:.2f} with "
                  f"alpha-generator={alpha_generator:.2f}: ")

            net = LightNet(dim=cfg.dim).to(device=cfg.device, dtype=cfg.dtype)
            generator = AlphaStableFlowLinear(net=net, dim=cfg.dim, alpha=alpha_generator,
                                              dtype=cfg.dtype, device=cfg.device,
                                              n_steps=cfg.n_steps)

            _, meta = train(generator, target_data=X_train.clone(), batch_size=cfg.batch_size,
                            n_epochs=cfg.n_epochs, lr=cfg.lr, use_adamw=cfg.use_adamw,
                            grad_clip_norm=cfg.grad_clip_norm, num_workers=cfg.num_workers,
                            device=cfg.device)

            m = 10.0 * msle(X_test.clone(), generator.sample(n_samples=cfg.n_samples))
            results[(float(alpha_data), float(alpha_generator))].append(m)
            print(f"[INFO] Evaluation: msle={m:.4f} (runtime: {time.perf_counter() - t0:.1f} s.)")

####################################################################################################
# Savings
filename = plot_heatmap(results=results, plot_dir="bench_2_figures", xlabel=r"$\alpha-$model",
                        ylabel=r"$\alpha-$data", fontsize=16)
print(f"[INFO] Saving '{filename}'")

####################################################################################################
# Timing
print(f"[INFO] Experiment runtime: {time.perf_counter() - t0_global:.1f} s.")
