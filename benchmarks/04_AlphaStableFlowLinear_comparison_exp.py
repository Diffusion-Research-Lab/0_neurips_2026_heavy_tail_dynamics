"""Benchmark 1 simulation: alpha-stable data model comparison."""

import argparse
import time
from pathlib import Path
from cauda.model import LightNet
from cauda.datasets import fetch_synthetic_data
from cauda.training import train
from cauda.metrics import msle, msle_90, msle_99
from cauda.flow import GaussianFlowLinear, AlphaStableFlowLinear
from labkit.config import load_config
from results_utils import create_run_dir, write_artifacts
from tqdm import tqdm


def compute_msle_metrics(x, x_ref):
    x_abs = x.abs()
    x_ref_abs = x_ref.abs()
    return {
        "MSLE": msle(x_abs, x_ref_abs),
        "MSLE_90": msle_90(x_abs, x_ref_abs),
        "MSLE_99": msle_99(x_abs, x_ref_abs),
    }


if __name__ == "__main__":
    t0_global = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config") / "bench_1_config.yaml")
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    args = parser.parse_args()

    cfg = load_config(args.config).set_up()
    run_dir = create_run_dir(args.out_root, "bench_1")
    print(f"[INFO] bench_1 config loaded: {args.config}")
    print(f"[INFO] bench_1 run directory: {run_dir}")
    print(f"[INFO] bench_1 alpha_data={float(cfg.alpha_data):.2f} alpha_model={float(cfg.alpha_model):.2f}")

    l_cls = [GaussianFlowLinear, AlphaStableFlowLinear]
    l_kwargs = [
        {"dim": cfg.dim, "dtype": cfg.dtype, "device": cfg.device, "n_steps": cfg.n_steps},
        {
            "dim": cfg.dim,
            "dtype": cfg.dtype,
            "device": cfg.device,
            "n_steps": cfg.n_steps,
            "alpha": float(cfg.alpha_model),
        },
    ]

    metric_names = ["MSLE", "MSLE_90", "MSLE_99"]
    raw = {(gen.__name__, metric): [] for gen in l_cls for metric in metric_names}

    for trial_idx in tqdm(range(cfg.n_trials), desc="bench1/trials", unit="trial"):
        print(f"[INFO] bench_1 trial {trial_idx + 1}/{cfg.n_trials}: generating synthetic data")
        x_train, _, x_test = fetch_synthetic_data(
            cfg.target_data_type,
            n_samples=cfg.n_samples,
            dim=cfg.dim,
            alpha=float(cfg.alpha_data),
            device=cfg.device,
            dtype=cfg.dtype,
        )

        for kwargs, gen_cls in tqdm(
            list(zip(l_kwargs, l_cls)),
            total=len(l_cls),
            desc=f"bench1/models t{trial_idx + 1}",
            leave=False,
            unit="model",
        ):
            print(f"[INFO] bench_1 trial {trial_idx + 1}: training {gen_cls.__name__}")
            net = LightNet(dim=cfg.dim).to(device=cfg.device, dtype=cfg.dtype)
            kwargs = dict(kwargs)
            kwargs["net"] = net
            generator = gen_cls(**kwargs)

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

            x_test_gen = generator.sample(n_samples=cfg.n_samples)
            metrics = compute_msle_metrics(x_test.clone(), x_test_gen)
            for metric_name, metric_value in metrics.items():
                raw[(gen_cls.__name__, metric_name)].append(float(10.0 * metric_value))

    results_payload = {
        "benchmark": "bench_1",
        "config_path": str(args.config),
        "metrics": [
            {
                "model": model,
                "metric": metric,
                "values": vals,
                "mean": float(sum(vals) / max(len(vals), 1)),
            }
            for (model, metric), vals in raw.items()
        ],
    }

    runtime = time.perf_counter() - t0_global
    run_txt = (
        f"benchmark: bench_1\n"
        f"config: {args.config}\n"
        f"run_dir: {run_dir}\n"
        f"n_trials: {cfg.n_trials}\n"
        f"runtime_sec: {runtime:.2f}\n"
    )

    write_artifacts(run_dir, cfg.as_dict(), results_payload, run_txt)
    print(f"[INFO] Saved run artifacts in {run_dir}")
