"""Benchmark 5 simulation: Barron-loss comparison across source/preprocessing choices."""

import argparse
import time
from pathlib import Path
from tqdm import tqdm
from genkit.datasets import fetch_synthetic_data
from genkit.metrics import metric_on_quantile, mssle, wasserstein_distance
from genkit.nn import MLPModel
from genkit.training import train
from genkit.visitor import CoreMetricsVisitor
from labkit.config import load_config
from _utils import build_model_specs, create_run_dir, summarize_metric_values, write_artifacts


BENCHMARK_NAME = "bench_5"


if __name__ == "__main__":
    t0_global = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("05_loss_H_comparison_benchmark_cfg.yml"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    args = parser.parse_args()

    cfg = load_config(args.config).set_up()
    run_dir = create_run_dir(args.out_root, BENCHMARK_NAME)
    print(f"[INFO] {BENCHMARK_NAME} config loaded: {args.config}")
    print(f"[INFO] {BENCHMARK_NAME} run directory: {run_dir}")

    metric_names = ["MSSLE_95", "WASS"]
    model_specs = build_model_specs(cfg)
    net_kwargs = {"width": cfg.width, "depth": cfg.depth}
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
        spec["name"]: {
            "meta": {
                "name": spec["name"],
                "source_distri": spec["source_distri"],
                "loss": spec["loss"],
                "loss_alpha": spec["loss_alpha"],
                "extra_kwargs": dict(spec["extra_kwargs"]),
            },
            "metrics": {metric: [] for metric in metric_names},
            "trials": [],
        }
        for spec in model_specs
    }

    for trial_idx in tqdm(range(cfg.n_trials), desc="bench5/trials", unit="trial"):
        print(f"[INFO] {BENCHMARK_NAME} trial {trial_idx + 1}/{cfg.n_trials}")

        x_train, _, x_test = fetch_synthetic_data(
            cfg.target_data_type,
            alpha=cfg.stable_alpha_data,
            n_samples=cfg.n_samples_train,
            dim=cfg.dim,
            device=cfg.device,
            dtype=cfg.fdtype,
        )
        x_test = x_test[: cfg.n_samples_test]

        for spec_idx, spec in enumerate(
            tqdm(model_specs, desc=f"bench5/models t{trial_idx + 1}", leave=False, unit="model")
        ):
            print(
                f"[INFO] {BENCHMARK_NAME} trial {trial_idx + 1}: "
                f"{spec['name']} ({spec_idx + 1}/{len(model_specs)})"
            )

            net = MLPModel(dim=cfg.dim, **net_kwargs).to(device=cfg.device, dtype=cfg.fdtype)
            generator = spec["cls"](
                net=net,
                dim=cfg.dim,
                n_steps=cfg.n_steps,
                device=cfg.device,
                fdtype=cfg.fdtype,
                idtype=cfg.idtype,
                **spec["extra_kwargs"],
            )

            t0 = time.perf_counter()
            generator, diagnostics = train(
                generative_model=generator,
                target_data=x_train.clone(),
                visitors=[CoreMetricsVisitor()],
                **train_kwargs,
            )
            print(f"[INFO] trained in {time.perf_counter() - t0:.1f}s")

            x_test_gen = generator.sample(n_samples=cfg.n_samples_test)
            metrics = {
                "MSSLE_95": float(metric_on_quantile(mssle, x_test.clone(), x_test_gen, xi=0.95)),
                "WASS": float(wasserstein_distance(x_test.clone(), x_test_gen)),
            }
            for metric_name, metric_value in metrics.items():
                results[spec["name"]]["metrics"][metric_name].append(metric_value)
            results[spec["name"]]["trials"].append(
                {
                    "trial_idx": trial_idx,
                    "metrics": metrics,
                    "diagnostics": {
                        "visitors": {
                            "core": {
                                "training_loss": diagnostics.get("visitors", {}).get("core", {}).get("training_loss", []),
                                "grad_variance_epoch": diagnostics.get("visitors", {}).get("core", {}).get("grad_variance_epoch", []),
                            }
                        }
                    },
                }
            )

    payload = {
        "benchmark": BENCHMARK_NAME,
        "config_path": str(args.config),
        "models": [
            {
                **entry["meta"],
                "metrics": {
                    metric_name: summarize_metric_values(values)
                    for metric_name, values in entry["metrics"].items()
                },
                "trials": entry["trials"],
            }
            for entry in results.values()
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
