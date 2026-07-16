"""Tests for benchmark helpers and entrypoints."""

from pathlib import Path
import importlib
import os
import pandas as pd
import subprocess
import sys
import pytest
import torch
import benchmarks._real_data_cache as real_data_cache
import benchmarks.utils as bench_utils
from genkit import GaussianFlowEDM

evaluate = importlib.import_module("benchmarks.02_evaluate")
pilot_analysis = importlib.import_module("benchmarks.03_pilot_analysis")
plotting_bench = importlib.import_module("benchmarks.04_plotting_bench")
imagenet128_viz_configs = importlib.import_module("benchmarks.06_make_imagenet128_viz")
imagenet128_viz = importlib.import_module("benchmarks.07_visualize_imagenet128")
compute_test_metrics = evaluate.compute_test_metrics
compute_test_vs_test_metrics = evaluate.compute_test_vs_test_metrics
sample_generator_in_batches = evaluate.sample_generator_in_batches

_main = importlib.import_module("benchmarks.01_main")
_resolved_config = _main._resolved_config
load_yaml = _main.load_yaml
require_section = _main.require_section
select_entries = _main.select_entries
build_dataset = _main.build_dataset
build_training_dataset = _main.build_training_dataset
build_network = _main.build_network
build_model = _main.build_model
resolve_data_device = _main.resolve_data_device

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREFETCH_SCRIPT = _PROJECT_ROOT / "scripts" / "prefetch.datasets.py"
_CONFIG_IMAGE_PILOT = _PROJECT_ROOT / "benchmarks" / "configs" / "pilot" / "image.yaml"
_PILOT_ANALYSIS_SCRIPT = _PROJECT_ROOT / "benchmarks" / "03_pilot_analysis.py"

_PILOT_FIXTURE_BATCHES = {
    "synth": "449824_02_alphastable_pilot_evaluate",
    "image": "449826_08_image_pilot_evaluate",
}
_PILOT_MODEL_NAMES = [
    "gaussian_flow_linear_euler",
    "gaussian_flow_linear_heun",
    "ddpm_v_ddpm",
    "ddpm_v_ddim",
    "dlpm_eps_a17",
    "dlpm_eps_a19",
    "tedm_origin_nu21",
    "tedm_origin_nu30",
]
_PILOT_MODEL_LABELS = {
    "gaussian_flow_linear_euler": "GF-Linear Euler",
    "gaussian_flow_linear_heun": "GF-Linear Heun",
    "ddpm_v_ddpm": "DDPM-V DDPM",
    "ddpm_v_ddim": "DDPM-V DDIM",
    "dlpm_eps_a17": "DLPM alpha=1.7",
    "dlpm_eps_a19": "DLPM alpha=1.9",
    "tedm_origin_nu21": "TEDM nu=2.1",
    "tedm_origin_nu30": "TEDM nu=3.0",
}


def _first_model_preset_by_name(models: dict) -> dict[str, str]:
    presets = {}
    for preset_name in models:
        for model_name in _PILOT_MODEL_NAMES:
            if preset_name.startswith(model_name) and model_name not in presets:
                presets[model_name] = preset_name
    return presets


def _write_fake_pilot_eval_artifacts(root: Path) -> None:
    pilot_root = _PROJECT_ROOT / "benchmarks" / "configs" / "pilot"
    dataset_rank = 0
    for family, batch_name in _PILOT_FIXTURE_BATCHES.items():
        config = load_yaml(pilot_root / f"{family}.yaml")
        batch_dir = root / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        model_presets = _first_model_preset_by_name(config["models"])
        train_preset = config["sweep"]["trains"][0]
        train_lr = float(config["trains"][train_preset]["lr"])
        for dataset_preset in config["sweep"]["datasets"]:
            for model_rank, model_name in enumerate(_PILOT_MODEL_NAMES):
                model_preset = model_presets[model_name]
                run_dir = batch_dir / f"{dataset_rank:03d}__{dataset_preset}__network__{model_preset}__{train_preset}"
                run_dir.mkdir(parents=True, exist_ok=True)
                rows = []
                for repeat_idx, offset in [(0, 0.0), (1, 0.05)]:
                    rows.append(
                        {
                            "source": "pilot_selection",
                            "metric_name": "validation_loss",
                            "value": float(1 + dataset_rank + model_rank + offset),
                            "checkpoint_epoch": 16,
                            "dataset_preset": dataset_preset,
                            "dataset_name": dataset_preset,
                            "model_preset": model_preset,
                            "model_name": model_name,
                            "model_label": _PILOT_MODEL_LABELS[model_name],
                            "train_preset": train_preset,
                            "train_lr": train_lr,
                            "eval_repeat_idx": repeat_idx,
                            "epoch": float("nan"),
                        }
                    )
                pd.DataFrame(rows).to_csv(run_dir / "scalars.csv.gz", index=False, compression="gzip")
                (run_dir / "summary.yaml").write_text("source_run_dir: synthetic-test-fixture\n", encoding="utf-8")
            dataset_rank += 1


@pytest.fixture(scope="module")
def generated_bench_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot_analysis")
    artifact_root = root / "artifacts"
    bench_config_root = root / "bench"
    report_root = root / "reports"
    _write_fake_pilot_eval_artifacts(artifact_root)
    subprocess.run(
        [
            sys.executable,
            str(_PILOT_ANALYSIS_SCRIPT),
            "--artifact-root",
            str(artifact_root),
            "--bench-config-root",
            str(bench_config_root),
            "--report-root",
            str(report_root),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{_PROJECT_ROOT}:{_PROJECT_ROOT / 'src'}"},
    )
    return {"bench_config_root": bench_config_root, "report_root": report_root}


def test_select_entries_keeps_literal_lists_out_of_grid():
    registry = {
        "unet": {
            "name": "unet",
            "params": {
                "channel_mult": {"literal": [1, 2, 4]},
                "width": [32, 64],
            },
        }
    }

    variants = select_entries("networks", registry, ["unet"])

    assert len(variants) == 2
    assert [variant["config"]["params"]["width"] for variant in variants] == [32, 64]
    assert all(variant["config"]["params"]["channel_mult"] == (1, 2, 4) for variant in variants)


def test_sample_generator_in_batches_respects_batch_size():
    class Generator:
        def __init__(self):
            self.calls = []

        def sample(self, n_samples):
            self.calls.append(n_samples)
            return torch.full((n_samples, 2), float(n_samples))

    generator = Generator()

    samples = sample_generator_in_batches(generator, n_samples=7, batch_size=3)

    assert generator.calls == [3, 3, 1]
    assert samples.shape == (7, 2)


def test_save_image_sample_grid_writes_png(tmp_path):
    images = torch.arange(6 * 4 * 4, dtype=torch.float32).reshape(6, 1, 4, 4)
    output_path = tmp_path / "samples.png"

    evaluate.save_image_sample_grid(
        images,
        output_path,
        title="test samples",
        normalize="per-image",
        n_cols=3,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_compute_test_metrics_keeps_other_metrics_when_mmd_fails(monkeypatch):
    def raise_mmd(*args, **kwargs):
        raise RuntimeError("mmd boom")

    monkeypatch.setattr(evaluate, "mmd_rbf", raise_mmd)
    x_ref = torch.arange(256, dtype=torch.float32).reshape(128, 2) + 1.0
    x_gen = x_ref + 0.1

    values, warnings = compute_test_metrics(
        x_ref,
        x_gen,
        max_mmd_samples=8,
    )

    assert values["MMD_RBF"] != values["MMD_RBF"]
    assert any("MMD_RBF_failed: RuntimeError: mmd boom" == warning for warning in warnings)
    for metric_name in [
        "TCE(90)",
        "TCE(99)",
        "TCE(99,9)",
        "TCE(99,99)",
    ]:
        assert torch.isfinite(torch.tensor(values[metric_name]))


def test_tail_coverage_grid_keeps_anchor_metrics():
    assert len(evaluate.TAIL_COVERAGE_METRICS) == 20
    for metric_name in ["TCE(90)", "TCE(99)", "TCE(99,9)", "TCE(99,99)"]:
        assert metric_name in evaluate.TAIL_COVERAGE_METRICS
    assert len(plotting_bench.TCE_METRICS) == 20


def test_compute_test_metrics_computes_mmd_for_high_dimensional_data():
    x_ref = torch.arange(12 * 32, dtype=torch.float32).reshape(12, 32)
    x_gen = x_ref + 0.1

    values, warnings = compute_test_metrics(x_ref, x_gen, max_mmd_samples=8)

    assert torch.isfinite(torch.tensor(values["MMD_RBF"]))
    assert not any("mmd_skipped" in warning.lower() for warning in warnings)


def test_compute_test_vs_test_metrics_computes_mmd_baseline():
    x_ref = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)

    values, warnings = compute_test_vs_test_metrics(x_ref, max_mmd_samples=8)

    assert torch.isfinite(torch.tensor(values["MMD_RBF"]))
    assert warnings == []


def test_selection_only_evaluation_estimates_validation_loss(tmp_path):
    batch_dir = tmp_path / "pilot_synth"
    run_dir = batch_dir / "001__alpha_stable_target__mlp__gaussian_flow_linear_euler_s2__pilot_lr1e4"
    run_dir.mkdir(parents=True)
    network_init = {
        "name": "mlp",
        "params": {
            "width": 8,
            "depth": 1,
            "time_dim": 8,
            "dropout": 0.0,
            "use_norm": False,
        },
    }
    net, _ = build_network(network_init, torch.zeros(2, 2))
    torch.save(
        {
            "model_init": {
                "network": network_init,
                "model": {
                    "name": "gaussian_flow_linear",
                    "params": {"n_steps": 4, "sigma_max": 2.0, "sampler": "euler"},
                },
                "dtype": "torch.float32",
                "device": "cpu",
                "train_shape": [16, 2],
            },
            "network_state_dict": net.state_dict(),
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "run:",
                "  dtype: float32",
                "  trial_idx: 0",
                "dataset:",
                "  kind: synthetic",
                "  name: alpha_stable",
                "  preset_name: alpha_stable_target",
                "  params:",
                "    n_samples: 24",
                "    alpha: 1.7",
                "    dim: 2",
                "  split:",
                "    val_size: 0.25",
                "    test_size: 0.25",
                "    random_state: 0",
                "    standardize: false",
                "network:",
                "  name: mlp",
                "  preset_name: mlp",
                "model:",
                "  name: gaussian_flow_linear",
                "  preset_name: gaussian_flow_linear_euler_s2",
                "  params:",
                "    sigma_max: 2.0",
                "train:",
                "  preset_name: pilot_lr1e4",
                "  n_epochs: 2",
                "  lr: 0.0001",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame({"epoch": [1, 2], "training_loss": [4.0, 3.0]}).to_csv(run_dir / "train_stats.csv", index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "benchmarks" / "02_evaluate.py"),
            "--batch-dir",
            str(batch_dir),
            "--selection-only",
            "--selection-repeats",
            "2",
            "--selection-batch-size",
            "4",
            "--device",
            "cpu",
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": f"{_PROJECT_ROOT}:{_PROJECT_ROOT / 'src'}"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    scalars = pd.read_csv(batch_dir.with_name("pilot_synth_evaluate") / run_dir.name / "scalars.csv.gz")
    selection = scalars[scalars["source"].eq("pilot_selection")]
    assert selection["metric_name"].unique().tolist() == ["validation_loss"]
    assert selection["eval_repeat_idx"].tolist() == [0.0, 1.0]
    assert torch.isfinite(torch.tensor(selection["value"].tolist())).all().item()


def test_image_class_recovery_metrics_count_support_and_histogram(monkeypatch):
    probe = {
        "eligible_classes": torch.tensor([0, 1]),
        "test_probs": torch.tensor([0.75, 0.25]),
        "test_acc": 0.9,
        "n_test_classes": 2,
    }
    x_gen = torch.zeros(4, 1, 8, 8)

    monkeypatch.setattr(evaluate, "predict_image_class_probe", lambda probe, x, device: torch.tensor([0, 0, 1, 2]))
    metrics = evaluate.compute_image_class_recovery_metrics(probe, x_gen, device="cpu")

    assert metrics["CLASS_RECOVERY_INDEX"] == pytest.approx(1.0)
    assert metrics["CLASS_HIST_TV"] == pytest.approx(0.25)
    assert metrics["N_TEST_CLASSES"] == 2.0
    assert metrics["N_GEN_CLASSES"] == 3.0
    assert metrics["CLASS_PROBE_TEST_ACC"] == pytest.approx(0.9)


def test_latest_config_batch_finds_run_name_directory(tmp_path):
    artifact_root = tmp_path / "benchmarks" / "artifacts"
    config_path = tmp_path / "benchmarks" / "configs" / "bench" / "synth" / "alpha_stable_iso" / "ddpm_v.yaml"
    config_path.parent.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    config_path.write_text("run:\n  name: bench_synth__alpha_stable_iso__ddpm_v\n", encoding="utf-8")
    expected = artifact_root / "20260527_173718_bench_synth__alpha_stable_iso__ddpm_v"
    expected.mkdir()
    (artifact_root / "20260527_173718_bench_synth__alpha_stable_iso__ddpm_v_evaluate").mkdir()

    assert bench_utils.latest_config_batch_dir(artifact_root, config_path) == expected


def test_latest_config_batch_falls_back_to_saved_config_metadata(tmp_path):
    artifact_root = tmp_path / "benchmarks" / "artifacts"
    config_path = tmp_path / "benchmarks" / "configs" / "bench" / "synth" / "alpha_stable_iso" / "ddpm_v.yaml"
    config_path.parent.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    config_path.write_text("run:\n  name: bench_synth__alpha_stable_iso__ddpm_v\n", encoding="utf-8")
    batch_dir = artifact_root / "1412454_ddpm_v"
    batch_dir.mkdir()
    (batch_dir / "summary_shard_000.txt").write_text("\n".join(["n_runs: 1", "config: /lustre/fswork/projects/rech/jcx/uor49lv/src/flowbench/benchmarks/configs/bench/synth/alpha_stable_iso/ddpm_v.yaml"]) + "\n", encoding="utf-8")  # noqa

    assert bench_utils.latest_config_batch_dir(artifact_root, config_path) == batch_dir


def test_latest_config_batch_falls_back_to_run_config(tmp_path):
    artifact_root = tmp_path / "benchmarks" / "artifacts"
    config_path = tmp_path / "benchmarks" / "configs" / "bench" / "synth" / "alpha_stable_iso" / "ddpm_v.yaml"
    config_path.parent.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    config_path.write_text("run:\n  name: bench_synth__alpha_stable_iso__ddpm_v\n", encoding="utf-8")
    batch_dir = artifact_root / "1412454_ddpm_v"
    run_dir = batch_dir / "001_alpha_stable_iso__mlp__ddpm_v__selected__trial-01"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run:\n  name: bench_synth__alpha_stable_iso__ddpm_v\n", encoding="utf-8")

    assert bench_utils.latest_config_batch_dir(artifact_root, config_path) == batch_dir


def test_analyze_pilot_ignores_benchmark_evaluation_batches(tmp_path):
    artifact_root = tmp_path / "artifacts"
    pilot_batch = artifact_root / "100_pilot_synth_evaluate"
    bench_batch = artifact_root / "999_bench_synth__alpha_stable_iso__ddpm_v_evaluate"
    pilot_batch.mkdir(parents=True)
    bench_batch.mkdir()

    assert pilot_analysis.discover_family_batch(artifact_root, "synth") == pilot_batch


def test_plotting_ignores_pilot_evaluation_batches(tmp_path):
    artifact_root = tmp_path / "artifacts"
    pilot_run = artifact_root / "100_pilot_synth_evaluate" / "001_alpha_stable_target"
    bench_failed_run = artifact_root / "999_ddpm_v_evaluate" / "001_failed_run"
    bench_run = artifact_root / "999_ddpm_v_evaluate" / "002_alpha_stable_target"
    pilot_run.mkdir(parents=True)
    bench_failed_run.mkdir(parents=True)
    bench_run.mkdir(parents=True)
    rows = [
        {
            "dataset_preset": "alpha_stable_target",
            "dataset_name": "alpha_stable",
            "model_label": "DDPM-V",
            "model_name": "ddpm_v",
            "source": "test_metrics",
            "metric_name": "MMD_RBF",
            "value": 1.0,
        }
    ]
    pd.DataFrame(rows).to_csv(pilot_run / "scalars.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(bench_run / "scalars.csv.gz", index=False, compression="gzip")

    batches = plotting_bench.discover_eval_batches(artifact_root)

    assert batches == {"alpha_stable_target": [bench_run.parent]}


def test_render_dataset_writes_mmd_and_tce_summary_figures(tmp_path):
    batch_dir = tmp_path / "artifacts" / "999_alpha_stable_target_evaluate"
    figure_root = tmp_path / "figures"
    table_root = tmp_path / "tables"
    metrics = {
        "TCE(90)": 0.11,
        "TCE(99)": 0.04,
        "TCE(99,9)": 0.02,
        "TCE(99,99)": 0.01,
    }

    for run_idx, (model_label, mmd_value) in enumerate([("GF-Linear", 0.12), ("DDPM-V", 0.2)], start=1):
        run_dir = batch_dir / f"{run_idx:03d}_alpha_stable_target"
        run_dir.mkdir(parents=True)
        rows = []
        for eval_repeat_idx in range(3):
            base_row = {
                "run_dir": run_dir.name,
                "checkpoint_epoch": 2,
                "dataset_preset": "alpha_stable_target",
                "dataset_name": "alpha_stable",
                "model_label": model_label,
                "model_name": model_label.lower(),
                "model_alpha": float("nan"),
                "eval_repeat_idx": eval_repeat_idx,
                "epoch": float("nan"),
            }
            rows.append(
                {
                    **base_row,
                    "source": "test_metrics",
                    "metric_name": "MMD_RBF",
                    "value": mmd_value + 0.01 * eval_repeat_idx,
                }
            )
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        **base_row,
                        "source": "test_metrics",
                        "metric_name": metric_name,
                        "value": value + 0.01 * eval_repeat_idx,
                    }
                )
        rows.append(
            {
                "run_dir": run_dir.name,
                "checkpoint_epoch": 2,
                "dataset_preset": "alpha_stable_target",
                "dataset_name": "alpha_stable",
                "model_label": model_label,
                "model_name": model_label.lower(),
                "model_alpha": float("nan"),
                "source": "test_vs_test_metrics",
                "metric_name": "MMD_RBF",
                "value": 0.05,
                "eval_repeat_idx": float("nan"),
                "epoch": float("nan"),
            }
        )
        pd.DataFrame(rows).to_csv(run_dir / "scalars.csv.gz", index=False, compression="gzip")

    stale_table = table_root / "alpha_stable_iso__performance.tex"
    stale_table.parent.mkdir(parents=True)
    stale_table.write_text("stale", encoding="utf-8")
    stale_figures = [
        figure_root / f"alpha_stable_iso__{stem}.pdf"
        for stem in ["mmd_rbf", "tce_90", "tce_95", "tce_99", "tce_999", "tce_9999"]
    ]
    for stale_figure in stale_figures:
        stale_figure.parent.mkdir(parents=True, exist_ok=True)
        stale_figure.write_text("stale", encoding="utf-8")

    plotting_bench.render_dataset(
        "alpha_stable_target",
        [batch_dir],
        table_root=table_root,
        figure_root=figure_root,
    )

    assert (figure_root / "alpha_stable_iso__mmd_rbf_boxplot.pdf").is_file()
    assert (figure_root / "alpha_stable_iso__tce_quantiles.pdf").is_file()
    assert not stale_table.exists()
    assert all(not path.exists() for path in stale_figures)


def test_plotting_writes_image_class_recovery_table(tmp_path):
    artifact_root = tmp_path / "artifacts"
    table_root = tmp_path / "tables"

    for dataset_name, metric_value in [("cifar100_lt", 0.25), ("imagenet_lt", 0.75)]:
        run_dir = artifact_root / f"999_{dataset_name}_evaluate" / f"001_{dataset_name}"
        run_dir.mkdir(parents=True)
        rows = [
            {
                "run_dir": run_dir.name,
                "checkpoint_epoch": 1,
                "dataset_preset": dataset_name,
                "dataset_name": dataset_name,
                "model_label": "DDPM-V",
                "model_name": "ddpm_v",
                "model_alpha": float("nan"),
                "source": "test_metrics",
                "metric_name": "CLASS_RECOVERY_INDEX",
                "value": 1.0,
            },
            {
                "run_dir": run_dir.name,
                "checkpoint_epoch": 2,
                "dataset_preset": dataset_name,
                "dataset_name": dataset_name,
                "model_label": "DDPM-V",
                "model_name": "ddpm_v",
                "model_alpha": float("nan"),
                "source": "test_metrics",
                "metric_name": "CLASS_HIST_TV",
                "value": 1.0 - metric_value,
            },
            {
                "run_dir": run_dir.name,
                "checkpoint_epoch": 2,
                "dataset_preset": dataset_name,
                "dataset_name": dataset_name,
                "model_label": "DDPM-V",
                "model_name": "ddpm_v",
                "model_alpha": float("nan"),
                "source": "test_metrics",
                "metric_name": "CLASS_RECOVERY_INDEX",
                "value": metric_value,
            },
        ]
        pd.DataFrame(rows).to_csv(run_dir / "scalars.csv.gz", index=False, compression="gzip")

    dataset_batches = plotting_bench.discover_eval_batches(artifact_root)
    plotting_bench.render_image_class_recovery_table(dataset_batches, table_root)

    table_path = table_root / "image_class_recovery.tex"
    contents = table_path.read_text(encoding="utf-8")
    assert "Image class recovery and class-histogram total variation" in contents
    assert "CIFAR100-LT" in contents
    assert "ImageNet-LT" in contents
    assert "Class Hist. TV" in contents


def test_image_bench_transformer_config_expands_to_one_network_variant():
    config = load_yaml(Path("benchmarks/configs/templates/image_bench.yaml"))
    variants = select_entries("networks", require_section(config, "networks"), ["transformer"])

    assert len(variants) == 1
    params = variants[0]["config"]["params"]
    assert params["n_steps"] == 256
    assert params["timestep_mode"] == "continuous"
    assert params["patch_size"] == 4
    assert params["num_layers"] == 12
    assert params["num_attention_heads"] == 6
    assert params["attention_head_dim"] == 64
    assert params["norm_type"] == "ada_norm_single"
    assert params["cross_attention_dim"] is None
    assert params["caption_channels"] is None
    assert params["use_additional_conditions"] is False


def test_build_network_supports_image_transformer():
    network_cfg = {
        "name": "transformer",
        "params": {
            "n_steps": 8,
            "patch_size": 2,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "timestep_mode": "continuous",
        },
    }
    net, params = build_network(network_cfg, torch.zeros(2, 3, 8, 8))

    assert net.__class__.__name__ == "TransformerModel"
    assert params["sample_size"] == 8
    assert params["in_channels"] == 3
    assert params["out_channels"] == 3


def test_build_network_rejects_nonsquare_image_transformer():
    network_cfg = {"name": "transformer", "params": {"n_steps": 8, "patch_size": 2}}

    with pytest.raises(ValueError, match="square image data"):
        build_network(network_cfg, torch.zeros(2, 3, 8, 12))


def test_build_model_registers_gaussian_flow_edm():
    net = torch.nn.Linear(2, 2)
    model, params = build_model(
        {"name": "gaussian_flow_edm", "params": {"n_steps": 4, "sigma_max": 1.0}},
        net,
        torch.zeros(3, 2),
        torch.float32,
        "cpu",
    )

    assert isinstance(model, GaussianFlowEDM)
    assert params["dim"] == 2
    assert params["fdtype"] == torch.float32
    assert params["device"] == "cpu"


def test_pilot_and_template_configs_have_required_sections_and_valid_sweeps():
    required_sections = {"run", "sweep", "datasets", "networks", "models", "trains", "save"}
    for path in sorted(Path("benchmarks/configs").rglob("*.yaml")):
        if "benchmarks/configs/bench/" in str(path):
            continue
        config = load_yaml(path)
        assert required_sections <= set(config), path.name
        sweep = require_section(config, "sweep")
        for section in ["datasets", "networks", "models", "trains"]:
            variants = select_entries(section, require_section(config, section), list(sweep[section]))
            assert variants, f"{path.name}: empty {section} sweep"


def test_image_pilot_caps_image_datasets_and_uncaps_hrrr():
    config = load_yaml(Path("benchmarks/configs/pilot/image.yaml"))

    assert "params" not in config["datasets"]["hrrr"]
    assert config["datasets"]["lvis"]["params"]["max_samples"] == 4096
    assert config["datasets"]["cifar100_lt"]["params"]["max_samples"] == 4096
    assert config["datasets"]["cifar100_lt"]["params"]["image_size"] == 32
    assert config["datasets"]["imagenet_lt"]["params"]["max_samples"] == 4096


def test_image_bench_uncaps_image_datasets_and_uses_target_resolutions():
    config = load_yaml(Path("benchmarks/configs/templates/image_bench.yaml"))
    assert config["datasets"]["lvis"]["params"]["image_size"] == 64
    assert config["datasets"]["cifar100_lt"]["params"]["image_size"] == 32
    assert config["datasets"]["imagenet_lt"]["params"]["image_size"] == 64
    assert "max_samples" not in config["datasets"]["lvis"]["params"]
    assert "max_samples" not in config["datasets"]["cifar100_lt"]["params"]
    assert "max_samples" not in config["datasets"]["imagenet_lt"]["params"]
    assert config["datasets"]["cifar100_lt"]["params"]["imbalance_factor"] == 10
    assert "category_frequency" not in config["datasets"]["lvis"]["params"]


def test_imagenet128_viz_config_sets_target_resolution_epochs_and_uncaps_dataset():
    source = {
        "run": {"name": "bench_image__imagenet_lt__ddpm_v", "dtype": "float32", "n_trial": 1},
        "selection": {"dataset_slug": "imagenet_lt", "selected_lr": 0.0002},
        "sweep": {"datasets": ["imagenet_lt"], "networks": ["transformer"], "models": ["ddpm_v"], "trains": ["selected"]},
        "datasets": {
            "imagenet_lt": {
                "kind": "real",
                "name": "imagenet_lt",
                "params": {"split": "train", "image_size": 64, "max_samples": 4096, "seed": 0},
                "split": {"val_size": 0.1, "test_size": 0.1, "random_state": 0, "standardize": False},
            },
        },
        "networks": {
            "transformer": {
                "name": "transformer",
                "params": {"patch_size": 4, "timestep_mode": "continuous", "sample_size": 64, "in_channels": 3, "out_channels": 3},
            }
        },
        "models": {"ddpm_v": {"name": "ddpm_v", "params": {"n_steps": 128, "sigma_max": 2.0}}},
        "trains": {"selected": {"device": "cuda", "batch_size": 8, "n_epochs": 128, "lr": 0.0002}},
        "save": {"root_dir": "benchmarks/artifacts"},
    }

    config = imagenet128_viz_configs.build_viz_config(source, "ddpm_v")

    assert config["run"]["name"] == "viz_image__imagenet_lt_96__ddpm_v"
    assert config["sweep"]["datasets"] == ["imagenet_lt_96"]
    assert config["sweep"]["models"] == ["ddpm_v"]
    dataset_params = config["datasets"]["imagenet_lt_96"]["params"]
    assert dataset_params["image_size"] == 96
    assert "max_samples" not in dataset_params
    assert "n_samples" not in dataset_params
    assert config["models"]["ddpm_v"]["params"]["sigma_max"] == 2.0
    assert config["trains"]["selected"]["lr"] == 0.0002
    assert config["trains"]["selected"]["n_epochs"] == 64
    assert "sample_size" not in config["networks"]["transformer"]["params"]
    assert "in_channels" not in config["networks"]["transformer"]["params"]
    assert "out_channels" not in config["networks"]["transformer"]["params"]


def test_imagenet128_viz_prepare_images_handles_multichannel_and_normalization():
    x = torch.stack([torch.linspace(-1, 2, steps=4 * 4).reshape(1, 4, 4).repeat(5, 1, 1)])

    clamped = imagenet128_viz.prepare_images(x, mode="clamp")
    normalized = imagenet128_viz.prepare_images(x, mode="per-image")

    assert clamped.shape == (1, 3, 4, 4)
    assert normalized.shape == (1, 3, 4, 4)
    assert float(clamped.min()) == 0.0
    assert float(clamped.max()) == 1.0
    assert torch.isclose(normalized.min(), torch.tensor(0.0))
    assert torch.isclose(normalized.max(), torch.tensor(1.0))


def test_nonblank_pilots_try_at_least_two_learning_rates():
    for filename in ["synth.yaml", "image.yaml"]:
        config = load_yaml(Path("benchmarks/configs/pilot") / filename)
        train_names = list(config["sweep"]["trains"])
        learning_rates = {float(config["trains"][name]["lr"]) for name in train_names}

        assert len(train_names) == 3
        assert len(learning_rates) == 3
        assert "pilot_lr1e3" not in config["trains"]


def test_pilot_configs_include_supported_sampler_variants():
    expected_counts = {"synth.yaml": (6, 6), "image.yaml": (4, 4)}
    for filename, (n_flow, n_ddpm) in expected_counts.items():
        config = load_yaml(Path("benchmarks/configs/pilot") / filename)
        models = config["models"]
        flow_names = [name for name in config["sweep"]["models"] if models[name]["name"] == "gaussian_flow_linear"]
        ddpm_names = [name for name in config["sweep"]["models"] if models[name]["name"] == "ddpm_v"]

        assert len(flow_names) == n_flow
        assert len(ddpm_names) == n_ddpm
        assert {models[name]["params"]["sampler"] for name in flow_names} == {"euler", "heun"}
        assert {models[name]["params"]["sampler"] for name in ddpm_names} == {"ddpm", "ddim"}
        assert all(models[name]["params"].get("eta", 0.0) == 0.0 for name in ddpm_names if models[name]["params"]["sampler"] == "ddim")
        dlpm_names = [name for name in config["sweep"]["models"] if models[name]["name"] == "dlpm_eps"]
        tedm_names = [name for name in config["sweep"]["models"] if models[name]["name"] == "tedm_origin"]
        assert {models[name]["params"]["alpha"] for name in dlpm_names} == {1.7, 1.9}
        assert {models[name]["params"]["nu"] for name in tedm_names} == {2.1, 3.0}


def test_generated_bench_config_tree_is_complete_and_explicit(generated_bench_outputs):
    bench_root = generated_bench_outputs["bench_config_root"]
    bench_configs = sorted(bench_root.rglob("*.yaml"))
    assert len(bench_configs) == 48
    assert bench_root / "synth/alpha_stable_iso/tedm_origin_nu21.yaml" in bench_configs
    assert bench_root / "synth/alpha_stable_iso/tedm_origin_nu30.yaml" in bench_configs
    assert bench_root / "image/lvis/dlpm_eps_a17.yaml" in bench_configs
    assert bench_root / "image/lvis/dlpm_eps_a19.yaml" in bench_configs
    assert bench_root / "image/lvis/gaussian_flow_linear_euler.yaml" in bench_configs


def test_generated_bench_configs_are_single_dataset_single_model_single_train(generated_bench_outputs):
    bench_root = generated_bench_outputs["bench_config_root"]
    for path in sorted(bench_root.rglob("*.yaml")):
        config = load_yaml(path)
        assert len(config["sweep"]["datasets"]) == 1
        assert len(config["sweep"]["networks"]) == 1
        assert len(config["sweep"]["models"]) == 1
        assert len(config["sweep"]["trains"]) == 1
        assert "selection" in config
        assert config["selection"]["selected_model_preset"]
        assert config["selection"]["selected_pilot_train_preset"]


def test_generated_bench_configs_use_template_budgets_but_selected_learning_rates(generated_bench_outputs):
    bench_root = generated_bench_outputs["bench_config_root"]
    synth_cfg = load_yaml(bench_root / "synth/alpha_stable_iso/gaussian_flow_linear_euler.yaml")
    image_cfg = load_yaml(bench_root / "image/lvis/tedm_origin_nu21.yaml")
    hrrr_cfg = load_yaml(bench_root / "image/hrrr/ddpm_v_ddpm.yaml")
    imagenet_cfg = load_yaml(bench_root / "image/imagenet_lt/dlpm_eps_a17.yaml")

    synth_train = synth_cfg["trains"][synth_cfg["sweep"]["trains"][0]]
    image_train = image_cfg["trains"][image_cfg["sweep"]["trains"][0]]
    hrrr_train = hrrr_cfg["trains"][hrrr_cfg["sweep"]["trains"][0]]
    imagenet_train = imagenet_cfg["trains"][imagenet_cfg["sweep"]["trains"][0]]

    assert synth_train["n_epochs"] == 512
    assert image_train["n_epochs"] == 128
    assert hrrr_train["n_epochs"] == 128
    assert imagenet_train["n_epochs"] == 128
    assert image_train["data_device"] == "cpu"
    assert hrrr_train["data_device"] == "cpu"
    assert imagenet_train["data_device"] == "cpu"
    assert image_train["pin_memory"] is True
    assert hrrr_train["pin_memory"] is True
    assert imagenet_train["pin_memory"] is True
    assert "data_device" not in synth_train
    assert synth_train["lr"] == synth_cfg["selection"]["selected_lr"]
    assert image_train["lr"] == image_cfg["selection"]["selected_lr"]
    assert hrrr_train["lr"] == hrrr_cfg["selection"]["selected_lr"]
    assert imagenet_train["lr"] == imagenet_cfg["selection"]["selected_lr"]
    assert synth_cfg["selection"]["selection_metric"] == "validation_loss"
    assert image_cfg["selection"]["selection_metric"] == "validation_loss"


def test_bench_templates_use_expected_trial_counts_and_single_train_preset():
    synth_cfg = load_yaml(Path("benchmarks/configs/templates/synth_bench.yaml"))
    image_cfg = load_yaml(Path("benchmarks/configs/templates/image_bench.yaml"))

    assert synth_cfg["run"]["n_trial"] == 5
    assert image_cfg["run"]["n_trial"] == 1
    assert synth_cfg["sweep"]["trains"] == ["standard"]
    assert image_cfg["sweep"]["trains"] == ["standard"]


def test_pilot_analysis_reports_exist_and_match_generated_configs(generated_bench_outputs):
    report_root = generated_bench_outputs["report_root"]
    report_csv = report_root / "pilot_best_settings.csv"
    report_md = report_root / "pilot_best_settings.md"
    report_png = report_root / "pilot_summary.png"
    assert report_csv.is_file()
    assert report_md.is_file()
    assert report_png.is_file()

    frame = pd.read_csv(report_csv)
    assert len(frame) == 48
    assert sorted(frame["dataset_slug"].unique().tolist()) == [
        "alpha_stable_iso",
        "alpha_stable_mix",
        "cifar100_lt",
        "hrrr",
        "imagenet_lt",
        "lvis",
    ]


def test_numbered_benchmark_entrypoints_work_from_outside_repo(tmp_path):
    repo_root = Path.cwd()
    for script in ["benchmarks/01_main.py", "benchmarks/02_evaluate.py"]:
        result = subprocess.run(
            [sys.executable, str(repo_root / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "usage:" in result.stdout


def test_resolve_data_device_defaults_to_training_device():
    assert resolve_data_device({"device": "cuda"}) == "cuda"
    assert resolve_data_device({"device": "cuda", "data_device": "cpu"}) == "cpu"
    assert resolve_data_device({}) == "cpu"


def test_image_pilot_config_explicitly_stages_heavy_data_on_cpu():
    config = load_yaml(_CONFIG_IMAGE_PILOT)

    for train_cfg in config["trains"].values():
        assert train_cfg["device"] == "cuda"
        assert train_cfg["data_device"] == "cpu"
        assert train_cfg["pin_memory"] is True


def test_preprocessed_real_dataset_cache_roundtrip(tmp_path, monkeypatch):
    dataset_cfg = {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 10, "max_samples": 4, "seed": 0},
        "split": {"val_size": 0.25, "test_size": 0.25, "random_state": 0, "standardize": False},
    }

    def fake_fetch_real_data(name, **kwargs):
        assert name == "cifar100_lt"
        assert kwargs["dtype"] is torch.float32
        assert kwargs["device"] == "cpu"
        assert kwargs["return_metadata"] is True
        return (
            torch.ones(2, 3),
            torch.full((1, 3), 2.0),
            torch.full((1, 3), 3.0),
            {"source": "test"},
        )

    monkeypatch.setattr(real_data_cache, "fetch_real_data", fake_fetch_real_data)

    result = real_data_cache.build_preprocessed_real_dataset(dataset_cfg, torch.float32, data_root=tmp_path)
    x_train, x_val, x_test = real_data_cache.load_preprocessed_real_dataset(dataset_cfg, torch.float32, data_root=tmp_path)
    x_train_only = real_data_cache.load_preprocessed_real_dataset(
        dataset_cfg, torch.float32, data_root=tmp_path, splits=("train",),
    )
    split_shapes = real_data_cache.load_preprocessed_real_dataset_shapes(dataset_cfg, torch.float32, data_root=tmp_path)

    assert result["status"] == "built"
    assert x_train.shape == (2, 3)
    assert len(x_train_only) == 1
    assert x_train_only[0].shape == (2, 3)
    assert split_shapes == {"train": (2, 3), "val": (1, 3), "test": (1, 3)}
    assert x_val.tolist() == [[2.0, 2.0, 2.0]]
    assert x_test.tolist() == [[3.0, 3.0, 3.0]]


def test_slurm_strict_real_dataset_loading_requires_processed_cache(tmp_path, monkeypatch):
    dataset_cfg = {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 10, "max_samples": 4, "seed": 0},
        "split": {"val_size": 0.25, "test_size": 0.25, "random_state": 0, "standardize": False},
    }

    monkeypatch.delenv("FLOWBENCH_DATA", raising=False)
    monkeypatch.setenv("FLOWBENCH_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA", "1")

    with pytest.raises(FileNotFoundError, match="Missing preprocessed real dataset cache"):
        build_dataset(dataset_cfg, torch.float32, "cpu")


def test_build_training_dataset_uses_train_split_and_cached_shapes(tmp_path, monkeypatch):
    dataset_cfg = _make_cifar100_lt_cfg()
    monkeypatch.setenv("FLOWBENCH_DATA", str(tmp_path))
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    real_data_cache.build_preprocessed_real_dataset(dataset_cfg, torch.float32, data_root=tmp_path)
    x_train, split_shapes = build_training_dataset(dataset_cfg, torch.float32, "cpu")

    assert x_train.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert split_shapes == {"train": (2, 3), "val": (1, 3), "test": (1, 3)}


def test_resolved_config_preserves_run_metadata(tmp_path):
    net = torch.nn.Linear(2, 2)
    x = torch.zeros(4, 2)
    config = _resolved_config(
        tmp_path / "config.yaml",
        torch.float32,
        tmp_path,
        {"name": "run_name", "seed": 12, "n_trial": 3},
        {"name": "alpha_stable", "preset_name": "dataset"},
        {"name": "mlp", "preset_name": "network"},
        {"name": "gaussian_flow_linear", "preset_name": "model"},
        {"preset_name": "train", "n_epochs": 2},
        {"root_dir": "_tmp"},
        2,
        x,
        {"train": tuple(x.shape), "val": tuple(x.shape), "test": tuple(x.shape)},
        net,
    )

    assert config["run"]["name"] == "run_name"
    assert config["run"]["seed"] == 12
    assert config["run"]["n_trial"] == 3
    assert config["run"]["trial_idx"] == 2
    assert config["run"]["resolved_dtype"] == "torch.float32"
    assert config["dataset"]["resolved_train_shape"] == [4, 2]
    assert config["dataset"]["resolved_val_shape"] == [4, 2]
    assert config["dataset"]["resolved_test_shape"] == [4, 2]


# ---------------------------------------------------------------------------
# Cache validation: SKIP/REBUILD logging and offline correctness
# ---------------------------------------------------------------------------


def _make_cifar100_lt_cfg():
    return {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 10, "max_samples": 4, "seed": 0},
        "split": {"val_size": 0.25, "test_size": 0.25, "random_state": 0, "standardize": False},
    }


def _write_imagenet_lt_cache_fixture(root: Path):
    from PIL import Image

    annotation_dir = root / "annotations"
    image_root = root / "imagenet"
    annotation_dir.mkdir(parents=True)
    image_root.mkdir(parents=True)
    records = [
        ("train/n00000001/img1.JPEG", 0, (255, 0, 0)),
        ("train/n00000001/img2.JPEG", 0, (220, 0, 0)),
        ("train/n00000002/img3.JPEG", 1, (0, 255, 0)),
        ("train/n00000002/img4.JPEG", 1, (0, 220, 0)),
        ("train/n00000003/img5.JPEG", 2, (0, 0, 255)),
        ("train/n00000003/img6.JPEG", 2, (0, 0, 220)),
    ]
    lines = []
    for relative_path, class_id, color in records:
        image_path = image_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 14), color=color).save(image_path)
        lines.append(f"{relative_path} {class_id}")
    (annotation_dir / "ImageNet_LT_train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, image_root


def _make_imagenet_lt_cfg(data_home: Path, image_root: Path):
    return {
        "kind": "real",
        "name": "imagenet_lt",
        "params": {
            "split": "train",
            "image_size": 8,
            "data_home": str(data_home),
            "imagenet_root": str(image_root),
            "seed": 0,
        },
        "split": {"val_size": 0.25, "test_size": 0.25, "random_state": 0, "standardize": False},
    }


def _fake_fetch_real_4samples(name, **kwargs):
    return (
        torch.ones(2, 3),
        torch.full((1, 3), 2.0),
        torch.full((1, 3), 3.0),
        {"source": "test"},
    )


def test_build_preprocessed_imagenet_lt_streams_directly_to_split_cache(tmp_path, monkeypatch):
    data_home, image_root = _write_imagenet_lt_cache_fixture(tmp_path / "imagenet_lt")
    cfg = _make_imagenet_lt_cfg(data_home, image_root)

    def boom(*args, **kwargs):
        raise AssertionError("ImageNet-LT prefetch should stream directly instead of calling fetch_real_data")

    monkeypatch.setattr(real_data_cache, "fetch_real_data", boom)

    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.bfloat16, data_root=tmp_path)
    x_train, x_val, x_test = real_data_cache.load_preprocessed_real_dataset(cfg, torch.bfloat16, data_root=tmp_path)
    metadata = real_data_cache.load_preprocessed_real_dataset_metadata(cfg, torch.bfloat16, data_root=tmp_path)

    assert result["status"] == "built"
    assert x_train.dtype == torch.bfloat16
    assert x_train.shape[1:] == (3, 8, 8)
    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == 6
    assert sum(metadata["splits"][name]["n_samples"] for name in ("train", "val", "test")) == 6
    assert all("records" in metadata["splits"][name] for name in ("train", "val", "test"))


def test_build_preprocessed_real_dataset_disables_image_loader_cache_by_default(tmp_path, monkeypatch):
    cfg = _make_cifar100_lt_cfg()
    seen = {}

    def fake_fetch_real_data(name, **kwargs):
        seen.update(kwargs)
        return _fake_fetch_real_4samples(name, **kwargs)

    monkeypatch.setattr(real_data_cache, "fetch_real_data", fake_fetch_real_data)

    real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)

    assert seen["cache"] is False
    assert "cache_dir" in seen


def test_build_preprocessed_real_dataset_skips_when_cache_is_valid(tmp_path, monkeypatch, capsys):
    cfg = _make_cifar100_lt_cfg()
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    capsys.readouterr()

    def boom(*args, **kwargs):
        raise AssertionError("fetch_real_data should not be called when cache is valid")

    monkeypatch.setattr(real_data_cache, "fetch_real_data", boom)
    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    out = capsys.readouterr().out

    assert result["status"] == "exists"
    assert "[SKIP] Valid preprocessed cache found for cifar100_lt" in out


def test_build_preprocessed_real_dataset_rebuilds_on_cache_key_mismatch(tmp_path, monkeypatch, capsys):
    cfg = _make_cifar100_lt_cfg()
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION,
            "cache_key": "BOGUS",
            "request": {"name": "cifar100_lt"},
            "x_train": torch.ones(2, 3),
            "x_val": torch.full((1, 3), 2.0),
            "x_test": torch.full((1, 3), 3.0),
        },
        cache_path,
    )
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    out = capsys.readouterr().out

    assert result["status"] == "built"
    assert "[REBUILD] cifar100_lt" in out
    assert "cache_key mismatch" in out


def test_build_preprocessed_real_dataset_rebuilds_on_cache_version_mismatch(tmp_path, monkeypatch, capsys):
    cfg = _make_cifar100_lt_cfg()
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = real_data_cache.real_dataset_cache_key(cfg, torch.float32)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION + 99,
            "cache_key": expected_key,
            "request": {"name": "cifar100_lt"},
            "x_train": torch.ones(2, 3),
            "x_val": torch.full((1, 3), 2.0),
            "x_test": torch.full((1, 3), 3.0),
        },
        cache_path,
    )
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    out = capsys.readouterr().out

    assert result["status"] == "built"
    assert "cache_version mismatch" in out


def test_build_preprocessed_real_dataset_rebuilds_when_cache_overshoots_requested_samples(tmp_path, monkeypatch, capsys):
    cfg = _make_cifar100_lt_cfg()
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = real_data_cache.real_dataset_cache_key(cfg, torch.float32)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION,
            "cache_key": expected_key,
            "request": {"name": "cifar100_lt"},
            "x_train": torch.zeros(40, 3),
            "x_val": torch.zeros(40, 3),
            "x_test": torch.zeros(40, 3),
        },
        cache_path,
    )
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    out = capsys.readouterr().out

    assert result["status"] == "built"
    assert "exceeds requested" in out


def test_build_preprocessed_real_dataset_rebuilds_when_tensor_missing(tmp_path, monkeypatch, capsys):
    cfg = _make_cifar100_lt_cfg()
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = real_data_cache.real_dataset_cache_key(cfg, torch.float32)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION,
            "cache_key": expected_key,
            "request": {"name": "cifar100_lt"},
            "x_train": torch.ones(2, 3),
            "x_val": torch.full((1, 3), 2.0),
            "x_test": "not a tensor",
        },
        cache_path,
    )
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    out = capsys.readouterr().out

    assert result["status"] == "built"
    assert "missing or invalid tensor 'x_test'" in out


def test_build_preprocessed_real_dataset_rebuilds_when_torch_load_fails(tmp_path, monkeypatch, capsys):
    cfg = _make_cifar100_lt_cfg()
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"not a torch payload")
    monkeypatch.setattr(real_data_cache, "fetch_real_data", _fake_fetch_real_4samples)

    result = real_data_cache.build_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)
    out = capsys.readouterr().out

    assert result["status"] == "built"
    assert "[REBUILD] cifar100_lt" in out
    assert "torch.load failed" in out


def test_load_preprocessed_real_dataset_rejects_invalid_cache(tmp_path):
    cfg = _make_cifar100_lt_cfg()
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION,
            "cache_key": "WRONG",
            "x_train": torch.ones(2, 3),
            "x_val": torch.ones(1, 3),
            "x_test": torch.ones(1, 3),
        },
        cache_path,
    )
    with pytest.raises(RuntimeError, match="cache_key mismatch"):
        real_data_cache.load_preprocessed_real_dataset(cfg, torch.float32, data_root=tmp_path)


def test_main_build_dataset_propagates_missing_cache_under_strict_mode(tmp_path, monkeypatch):
    cfg = _make_cifar100_lt_cfg()
    monkeypatch.delenv("FLOWBENCH_DATA", raising=False)
    monkeypatch.setenv("FLOWBENCH_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA", "1")

    with pytest.raises(FileNotFoundError, match="Missing preprocessed real dataset cache"):
        build_dataset(cfg, torch.float32, "cpu")


# ---------------------------------------------------------------------------
# prefetch.datasets.py --check-only smoke tests (exec via subprocess)
# ---------------------------------------------------------------------------


def _run_precheck(data_root: Path, args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "FLOWBENCH_DATA": str(data_root), "PYTHONPATH": f"{_PROJECT_ROOT}:{_PROJECT_ROOT / 'src'}"}
    return subprocess.run(
        [sys.executable, str(_PREFETCH_SCRIPT), "--check-only", *args],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_prefetch_check_only_exits_2_when_cache_missing(tmp_path):
    result = _run_precheck(tmp_path, ["--only-dataset", "cifar100_lt", str(_CONFIG_IMAGE_PILOT)])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "[MISSING]" in result.stdout
    assert "cifar100_lt" in result.stdout


def test_prefetch_check_only_exits_0_when_cache_valid(tmp_path):
    cfg_dict = real_data_cache.real_dataset_request(
        {
            "kind": "real",
            "name": "cifar100_lt",
            "params": {"split": "train", "image_size": 32, "imbalance_factor": 10, "max_samples": 4096, "seed": 0},
            "split": {"val_size": 0.1, "test_size": 0.1, "random_state": 0, "standardize": False},
        },
        torch.bfloat16,
    )
    cfg = {"kind": "real", "name": "cifar100_lt", "params": cfg_dict["params"], "split": cfg_dict["split"]}
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.bfloat16, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = real_data_cache.real_dataset_cache_key(cfg, torch.bfloat16)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION,
            "cache_key": expected_key,
            "request": {"name": "cifar100_lt"},
            "x_train": torch.zeros(8, 3, 32, 32, dtype=torch.bfloat16),
            "x_val": torch.zeros(4, 3, 32, 32, dtype=torch.bfloat16),
            "x_test": torch.zeros(4, 3, 32, 32, dtype=torch.bfloat16),
        },
        cache_path,
    )

    result = _run_precheck(tmp_path, ["--only-dataset", "cifar100_lt", str(_CONFIG_IMAGE_PILOT)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[VALID]" in result.stdout
    assert "no sbatch needed" in result.stdout


def test_prefetch_check_only_exits_2_when_cache_for_other_config(tmp_path):
    """A cache built for a different (params/seed/dtype) config lives at a different
    filename, so the precheck must report MISSING for the requested config and exit 2."""
    other_cfg = {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 10, "max_samples": 99999, "seed": 0},
        "split": {"val_size": 0.1, "test_size": 0.1, "random_state": 0, "standardize": False},
    }
    other_cache_path = real_data_cache.real_dataset_cache_path(other_cfg, torch.float32, data_root=tmp_path)
    other_cache_path.parent.mkdir(parents=True, exist_ok=True)
    other_cache_path.write_bytes(b"unrelated cache")

    result = _run_precheck(tmp_path, ["--only-dataset", "cifar100_lt", str(_CONFIG_IMAGE_PILOT)])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "[MISSING]" in result.stdout
    assert "cifar100_lt" in result.stdout


def test_prefetch_torch_free_cache_key_matches_runtime():
    """The login-node precheck must produce identical cache keys to the runtime builder."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prefetch_datasets", _PREFETCH_SCRIPT
    )
    prefetch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefetch_module)
    cfgs = [
        {
            "kind": "real",
            "name": "cifar100_lt",
            "params": {"split": "train", "image_size": 64, "imbalance_factor": 10, "max_samples": 256, "seed": 0},
            "split": {"val_size": 0.1, "test_size": 0.1, "random_state": 0, "standardize": False},
        },
        {
            "kind": "real",
            "name": "imagenet_lt",
            "params": {"split": "train", "image_size": 64, "max_samples": 50000, "seed": 0},
            "split": {"val_size": 0.1, "test_size": 0.1, "random_state": 0, "standardize": False},
        },
    ]
    assert prefetch_module.CACHE_VERSION == real_data_cache.CACHE_VERSION
    for dtype, dtype_str in [(torch.float32, "float32"), (torch.float64, "float64"), (torch.bfloat16, "bfloat16")]:
        for cfg in cfgs:
            runtime_key = real_data_cache.real_dataset_cache_key(cfg, dtype)
            yaml_key = prefetch_module.cache_key_from_yaml(cfg, dtype_str)
            assert runtime_key == yaml_key, (cfg["name"], dtype_str, runtime_key, yaml_key)


def test_prefetch_skips_missing_generated_config_directories(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prefetch_datasets", _PREFETCH_SCRIPT
    )
    prefetch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefetch_module)

    missing_dir = tmp_path / "bench"
    missing_file = tmp_path / "missing.yaml"

    assert prefetch_module._expand_config_inputs([missing_dir]) == []
    assert prefetch_module._expand_config_inputs([missing_file]) == [missing_file]


def test_prefetch_check_only_path_does_not_import_torch():
    """Loading scripts/prefetch.datasets.py must not import torch (login-node guarantee)."""
    code = (
        "import sys, builtins\n"
        "_orig_import = builtins.__import__\n"
        "def guard(name, *a, **kw):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        raise RuntimeError(f'torch import attempted via {name!r}')\n"
        "    return _orig_import(name, *a, **kw)\n"
        "builtins.__import__ = guard\n"
        f"import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('prefetch_datasets', r'{_PREFETCH_SCRIPT}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": f"{_PROJECT_ROOT}:{_PROJECT_ROOT / 'src'}"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
