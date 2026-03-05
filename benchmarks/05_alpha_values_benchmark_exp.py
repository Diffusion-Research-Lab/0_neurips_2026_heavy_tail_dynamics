"""Benchmark 2 simulation: alpha-data vs alpha-model grid."""

import argparse
import time
from pathlib import Path
from cauda.model import LightNet
from cauda.datasets import fetch_synthetic_data
from cauda.training import train
from cauda.metrics import msle_at_quantile
from cauda.flow import AlphaStableFlowLinear
from labkit.config import load_config
from results_utils import create_run_dir, write_artifacts
from tqdm import tqdm


def msle(x, x_ref):
    return msle_at_quantile(x.abs(), x_ref.abs())


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

    results = {
        (float(a_d), float(a_g)): []
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
                dtype=cfg.dtype,
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
                net = LightNet(dim=cfg.dim).to(device=cfg.device, dtype=cfg.dtype)
                generator = AlphaStableFlowLinear(
                    net=net,
                    dim=cfg.dim,
                    alpha=alpha_generator,
                    dtype=cfg.dtype,
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

                m = 10.0 * msle(x_test.clone(), generator.sample(n_samples=cfg.n_samples))
                results[(float(alpha_data), float(alpha_generator))].append(float(m))

    payload = {
        "benchmark": "bench_2",
        "config_path": str(args.config),
        "grid": [
            {
                "alpha_data": a_d,
                "alpha_model": a_g,
                "values": vals,
                "mean": float(sum(vals) / max(len(vals), 1)),
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
