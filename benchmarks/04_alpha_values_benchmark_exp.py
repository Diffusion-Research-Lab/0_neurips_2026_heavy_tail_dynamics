"""Benchmark 2 simulation: alpha-data vs alpha-model grid."""

import argparse
import time
from pathlib import Path
from genkit.nn import MLPModel
from genkit.datasets import fetch_synthetic_data
from genkit.training import train
from genkit.metrics import mssle_90, mssle_95
from genkit.flow import AlphaStableFlowLinear
from labkit.config import load_config
from _utils import create_run_dir, write_artifacts
from tqdm import tqdm


def compute_mssle_metrics(x, x_ref):
    x_abs = x.abs()
    x_ref_abs = x_ref.abs()
    return {
        "MSSLE_90": mssle_90(x_abs, x_ref_abs),
        "MSSLE_95": mssle_95(x_abs, x_ref_abs),
    }


if __name__ == "__main__":
    t0_global = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config") / "bench_2_config.yaml")
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    args = parser.parse_args()

    cfg = load_config(args.config).set_up()
    run_dir = create_run_dir(args.out_root, "bench_2")
    print(f"[INFO] bench_2 config loaded: {args.config}")
    print(f"[INFO] bench_2 run directory: {run_dir}")

    metric_names = ["MSSLE", "MSSLE_90", "MSSLE_95"]
    results = {
        (float(a_d), float(a_g)): {metric: [] for metric in metric_names}
        for a_d in cfg.l_alpha_data
        for a_g in cfg.l_alpha_generator
    }

    for trial_idx in tqdm(range(cfg.n_trials), desc="bench2/trials", unit="trial"):
        print(f"[INFO] bench_2 trial {trial_idx + 1}/{cfg.n_trials}")
        for alpha_data in tqdm(
            cfg.l_alpha_data,
            desc=f"bench2/alpha_data t{trial_idx + 1}",
            leave=False,
            unit="alpha",
        ):
            print(f"[INFO] bench_2 trial {trial_idx + 1}: alpha_data={float(alpha_data):.3f}")
            x_train, _, x_test = fetch_synthetic_data(
                cfg.target_data_type,
                n_samples=cfg.n_samples,
                dim=cfg.dim,
                alpha=alpha_data,
                device=cfg.device,
                dtype=cfg.fdtype,
            )

            for alpha_generator in tqdm(
                cfg.l_alpha_generator,
                desc=f"bench2/alpha_model d{float(alpha_data):.3f}",
                leave=False,
                unit="alpha",
            ):
                print(
                    f"[INFO] bench_2 trial {trial_idx + 1}: "
                    f"train AlphaStableFlowLinear(alpha_model={float(alpha_generator):.3f})"
                )
                net = MLPModel(dim=cfg.dim).to(device=cfg.device, dtype=cfg.fdtype)
                generator = AlphaStableFlowLinear(
                    net=net,
                    dim=cfg.dim,
                    alpha=alpha_generator,
                    fdtype=cfg.fdtype,
                    idtype=cfg.idtype,
                    device=cfg.device,
                    n_steps=cfg.n_steps,
                )

                train(
                    generator,
                    target_data=x_train.clone(),
                    batch_size=cfg.batch_size,
                    n_epochs=cfg.n_epochs,
                    lr=cfg.lr,
                    use_adamw=cfg.use_adamw,
                    grad_clip_norm=cfg.grad_clip_norm,
                    num_workers=cfg.num_workers,
                    device=cfg.device,
                )

                metrics = compute_mssle_metrics(
                    x_test.clone(),
                    generator.sample(n_samples=cfg.n_samples),
                )
                for metric_name, metric_value in metrics.items():
                    results[(float(alpha_data), float(alpha_generator))][metric_name].append(
                        float(10.0 * metric_value)
                    )

    payload = {
        "benchmark": "bench_2",
        "config_path": str(args.config),
        "grid": [
            {
                "alpha_data": a_d,
                "alpha_model": a_g,
                "metrics": {
                    metric_name: {
                        "values": vals[metric_name],
                        "mean": float(sum(vals[metric_name]) / max(len(vals[metric_name]), 1)),
                    }
                    for metric_name in metric_names
                },
            }
            for (a_d, a_g), vals in results.items()
        ],
    }

    runtime = time.perf_counter() - t0_global
    run_txt = (
        f"benchmark: bench_2\n"
        f"config: {args.config}\n"
        f"run_dir: {run_dir}\n"
        f"n_trials: {cfg.n_trials}\n"
        f"runtime_sec: {runtime:.2f}\n"
    )

    write_artifacts(run_dir, cfg.as_dict(), payload, run_txt)
    print(f"[INFO] Saved run artifacts in {run_dir}")
