"""Heavy tail illustrative example."""

import argparse
import pprint
import time
from pathlib import Path
from cauda.model import LightNet
from cauda.datasets import fetch_synthetic_data
from cauda.training import train
from cauda.metrics import msle_at_quantile
from cauda.table import save_double_entry_table
from cauda.flow import GaussianFlowLinear, AlphaStableFlowLinear
from labkit.config import load_config


####################################################################################################
# Settings
t0_global = time.perf_counter()

config_dir = Path("configs")
default_cfg = config_dir / "bench_1_config.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=default_cfg)
args = parser.parse_args()

cfg = load_config(args.config).set_up()

print('-' * 40)
print("Experimental configuration:")
print("---------------------------")
pprint.pprint(cfg.as_dict())
print('-' * 40)


def msle(x, x_ref):
    return msle_at_quantile(x.abs(), x_ref.abs())


l_cls = [GaussianFlowLinear, AlphaStableFlowLinear]
l_kwargs = [{'dim': cfg.dim, 'dtype': cfg.dtype, 'device': cfg.device, 'n_steps': cfg.n_steps},
            {'dim': cfg.dim, 'dtype': cfg.dtype, 'device': cfg.device, 'n_steps': cfg.n_steps, 'alpha': 1.5},
            ]
l_metrics = [msle,]
l_coefs = [10.0,]

results = dict()
for gen_cls in l_cls:
    for metric_func in l_metrics:
        results[(str(gen_cls.__name__), str(metric_func.__name__).upper())] = []

####################################################################################################
# Main
for n in range(cfg.n_trials):

    X_train, _, X_test = fetch_synthetic_data(cfg.target_data_type, n_samples=cfg.n_samples,
                                              dim=cfg.dim, alpha=cfg.alpha, device=cfg.device,
                                              dtype=cfg.dtype)

    print(f"[INFO] Running trial {n + 1:02d} / {cfg.n_trials:02d}")

    for kwargs, gen_cls in zip(l_kwargs, l_cls):

        t0 = time.perf_counter()

        print(f"[INFO] Running experiment on '{cfg.target_data_type}' data with "
              f"'{gen_cls.__name__}' model: ")

        net = LightNet(dim=cfg.dim).to(device=cfg.device, dtype=cfg.dtype)
        kwargs['net'] = net
        generator = gen_cls(**kwargs)

        _, meta = train(generator, target_data=X_train.clone(), batch_size=cfg.batch_size,
                        n_epochs=cfg.n_epochs, lr=cfg.lr, use_adamw=cfg.use_adamw,
                        grad_clip_norm=cfg.grad_clip_norm, num_workers=cfg.num_workers,
                        device=cfg.device)

        print("[INFO] Evaluation:", end="")
        X_test_gen = generator.sample(n_samples=cfg.n_samples)
        for coef, metric_func in zip(l_coefs, l_metrics):
            m = coef * metric_func(X_test.clone(), X_test_gen)
            results[(str(gen_cls.__name__), str(metric_func.__name__).upper())].append(m)
            print(f" {metric_func.__name__} = {m:.4f},", end="")

        print(f" (runtime: {time.perf_counter() - t0:.1f} s.)")

####################################################################################################
# Savings
col_order = ['GaussianFlowLinear', 'AlphaStableFlowLinear']
col_name = {'GaussianFlowLinear': 'Lipman et al',
            'AlphaStableFlowLinear': 'Our approach',
            }
metric_name = {'MSLE': r'$10 \times \mathrm{MSLE}_{\xi=0.95}$',
               }
caption = (r"Comparison of generative models on $\alpha$-stable synthetic data ($\alpha = "
           f"{cfg.alpha:.2f}$).")
filename = save_double_entry_table(results=results,
                                   plot_dir="_tables",
                                   fmt="{:.2f}",
                                   caption=caption,
                                   label=f"tab:synthetic_alpha_{cfg.alpha:.2f}",
                                   col_order=col_order,
                                   metric_direction={"MSLE": "down"},
                                   col_name=col_name,
                                   metric_name=metric_name,
                                   show_metric_arrows=True,
                                   bold_best_in_row=True,
                                   suffix=f"_{cfg.target_data_type}_{gen_cls.__name__}",
                                   )
print(f"[INFO] Saving '{filename}'")

####################################################################################################
# Timing
print(f"[INFO] Experiment runtime: {time.perf_counter() - t0_global:.1f} s.")
