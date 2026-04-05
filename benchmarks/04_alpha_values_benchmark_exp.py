"""Benchmark 2 simulation: alpha-data vs alpha-model grid."""

import argparse
import time
from pathlib import Path
from tqdm import tqdm
from genkit.datasets import fetch_synthetic_data
from genkit.flow import AlphaStableFlowLinear
from genkit.metrics import metric_on_quantile, mssle
from genkit.nn import MLPModel
from genkit.training import train
from labkit.config import load_config
from _utils import create_run_dir, write_artifacts


BENCHMARK_NAME = "bench_2"


if __name__ == "__main__":
    t0_global = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("04_alpha_values_benchmark_cfg.yml"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    args = parser.parse_args()

    cfg = load_config(args.config).set_up()
    run_dir = create_run_dir(args.out_root, BENCHMARK_NAME)
    print(f"[INFO] {BENCHMARK_NAME} config loaded: {args.config}")
    print(f"[INFO] {BENCHMARK_NAME} run directory: {run_dir}")

    metric_names = ["MSSLE_95"]
    train_kwargs = {
        "batch_size": cfg.batch_size,
        "n_epochs": cfg.n_epochs,
        "lr": cfg.lr,
        "use_adamw": cfg.use_adamw,
        "grad_clip_norm": cfg.grad_clip_norm,
        "num_workers": cfg.num_workers,
        "device": cfg.device,
    }

    results = {
        (float(alpha_data), float(alpha_model)): {metric: [] for metric in metric_names}
        for alpha_data in cfg.l_alpha_data
        for alpha_model in cfg.l_alpha_generator
    }

    for trial_idx in tqdm(range(cfg.n_trials), desc="bench2/trials", unit="trial"):
        print(f"[INFO] {BENCHMARK_NAME} trial {trial_idx + 1}/{cfg.n_trials}")

        for alpha_data in tqdm(
            cfg.l_alpha_data,
            desc=f"bench2/alpha_data t{trial_idx + 1}",
            leave=False,
            unit="alpha",
        ):
            alpha_data = float(alpha_data)
            print(f"[INFO] {BENCHMARK_NAME} trial {trial_idx + 1}: alpha_data={alpha_data:.3f}")

            x_train, _, x_test = fetch_synthetic_data(
                cfg.target_data_type,
                n_samples=cfg.n_samples,
                dim=cfg.dim,
                alpha=alpha_data,
                device=cfg.device,
                dtype=cfg.fdtype,
            )

            for alpha_model in tqdm(
                cfg.l_alpha_generator,
                desc=f"bench2/alpha_model d{alpha_data:.3f}",
                leave=False,
                unit="alpha",
            ):
                alpha_model = float(alpha_model)
                print(
                    f"[INFO] {BENCHMARK_NAME} trial {trial_idx + 1}: "
                    f"train AlphaStableFlowLinear(alpha_model={alpha_model:.3f})"
                )

                net = MLPModel(dim=cfg.dim).to(device=cfg.device, dtype=cfg.fdtype)
                generator = AlphaStableFlowLinear(
                    net=net,
                    dim=cfg.dim,
                    alpha=alpha_model,
                    fdtype=cfg.fdtype,
                    idtype=cfg.idtype,
                    device=cfg.device,
                    n_steps=cfg.n_steps,
                )

                generator, _ = train(
                    generative_model=generator,
                    target_data=x_train.clone(),
                    **train_kwargs,
                )

                x_test_gen = generator.sample(n_samples=cfg.n_samples)
                metrics = {
                    "MSSLE_95": float(metric_on_quantile(mssle, x_test.clone(), x_test_gen, xi=0.95)),
                }
                for metric_name, metric_value in metrics.items():
                    results[(alpha_data, alpha_model)][metric_name].append(metric_value)

    payload = {
        "benchmark": BENCHMARK_NAME,
        "config_path": str(args.config),
        "grid": [
            {
                "alpha_data": alpha_data,
                "alpha_model": alpha_model,
                "metrics": {
                    metric_name: {
                        "values": values[metric_name],
                        "mean": float(sum(values[metric_name]) / max(len(values[metric_name]), 1)),
                    }
                    for metric_name in metric_names
                },
            }
            for (alpha_data, alpha_model), values in results.items()
        ],
    }

    runtime = time.perf_counter() - t0_global
    run_txt = (
        f"benchmark: {BENCHMARK_NAME}\n"
        f"config: {args.config}\n"
        f"run_dir: {run_dir}\n"
        f"n_trials: {cfg.n_trials}\n"
        f"runtime_sec: {runtime:.2f}\n"
    )

    write_artifacts(run_dir, cfg.as_dict(), payload, run_txt)
    print(f"[INFO] Saved run artifacts in {run_dir}")
