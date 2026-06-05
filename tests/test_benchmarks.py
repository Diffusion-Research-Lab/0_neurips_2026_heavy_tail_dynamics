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

evaluate = importlib.import_module("benchmarks.02_evaluate")
pilot_analysis = importlib.import_module("benchmarks.03_pilot_analysis")
plotting_bench = importlib.import_module("benchmarks.04_plotting_bench")
compute_test_metrics = evaluate.compute_test_metrics
compute_test_vs_test_metrics = evaluate.compute_test_vs_test_metrics
sample_generator_in_batches = evaluate.sample_generator_in_batches

_main = importlib.import_module("benchmarks.01_main")
_resolved_config = _main._resolved_config
load_yaml = _main.load_yaml
require_section = _main.require_section
select_entries = _main.select_entries
build_dataset = _main.build_dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREFETCH_SCRIPT = _PROJECT_ROOT / "scripts" / "prefetch.datasets.py"
_CONFIG_IMAGE_PILOT = _PROJECT_ROOT / "benchmarks" / "configs" / "pilot" / "image.yaml"
_PILOT_ANALYSIS_SCRIPT = _PROJECT_ROOT / "benchmarks" / "03_pilot_analysis.py"

_PILOT_FIXTURE_BATCHES = {
    "synth": "449824_02_alphastable_pilot_evaluate",
    "image": "449826_08_image_pilot_evaluate",
}
_PILOT_FIXTURE_METRICS = ["INNER_LOSS_VAL"]


def _first_model_preset_by_name(models: dict) -> dict[str, str]:
    presets = {}
    for preset_name in models:
        for model_name in ["gaussian_flow_ot", "gaussian_flow_linear", "ddpm_v", "dlpm_eps", "tedm_origin"]:
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
            for model_rank, model_name in enumerate(["gaussian_flow_ot", "gaussian_flow_linear", "ddpm_v", "dlpm_eps", "tedm_origin"]):
                model_preset = model_presets[model_name]
                run_dir = batch_dir / f"{dataset_rank:03d}__{dataset_preset}__network__{model_preset}__{train_preset}"
                run_dir.mkdir(parents=True, exist_ok=True)
                rows = []
                for metric_rank, metric_name in enumerate(_PILOT_FIXTURE_METRICS):
                    rows.append(
                        {
                            "source": "pilot_selection",
                            "metric_name": metric_name,
                            "value": float(1 + dataset_rank + model_rank + metric_rank / 10.0),
                            "checkpoint_epoch": 16,
                            "dataset_preset": dataset_preset,
                            "dataset_name": dataset_preset,
                            "model_preset": model_preset,
                            "model_name": model_name,
                            "model_label": {
                                "gaussian_flow_ot": "GF-OT",
                                "gaussian_flow_linear": "GF-Linear",
                                "ddpm_v": "DDPM-V",
                                "dlpm_eps": "DLPM",
                                "tedm_origin": "TEDM-Orig",
                            }[model_name],
                            "train_preset": train_preset,
                            "train_lr": train_lr,
                            "eval_repeat_idx": 0,
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
    bench_run = artifact_root / "999_ddpm_v_evaluate" / "001_alpha_stable_target"
    pilot_run.mkdir(parents=True)
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

    for run_idx, (model_label, mmd_value) in enumerate([("GF-OT", 0.12), ("DDPM-V", 0.2)], start=1):
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


def test_image_bench_unet_config_expands_to_one_network_variant():
    config = load_yaml(Path("benchmarks/configs/templates/image_bench.yaml"))
    variants = select_entries("networks", require_section(config, "networks"), ["unet"])

    assert len(variants) == 1
    params = variants[0]["config"]["params"]
    assert params["attention_resolutions"] == (4,)
    assert params["channel_mult"] == (1, 2, 4)


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


def test_image_pilot_caps_real_datasets_and_uses_native_cifar_resolution():
    config = load_yaml(Path("benchmarks/configs/pilot/image.yaml"))

    assert config["datasets"]["hrrr"]["params"]["n_samples"] == 4096
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
    assert config["datasets"]["cifar100_lt"]["params"]["imbalance_factor"] == 20
    assert "category_frequency" not in config["datasets"]["lvis"]["params"]


def test_nonblank_pilots_try_at_least_two_learning_rates():
    for filename in ["synth.yaml", "image.yaml"]:
        config = load_yaml(Path("benchmarks/configs/pilot") / filename)
        train_names = list(config["sweep"]["trains"])
        learning_rates = {float(config["trains"][name]["lr"]) for name in train_names}

        assert len(train_names) >= 2
        assert len(learning_rates) >= 2
        assert "pilot_lr1e3" not in config["trains"]


def test_generated_bench_config_tree_is_complete_and_explicit(generated_bench_outputs):
    bench_root = generated_bench_outputs["bench_config_root"]
    bench_configs = sorted(bench_root.rglob("*.yaml"))
    assert len(bench_configs) == 30
    assert bench_root / "synth/alpha_stable_iso/tedm_origin.yaml" in bench_configs
    assert bench_root / "image/lvis/gaussian_flow_ot.yaml" in bench_configs


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
    synth_cfg = load_yaml(bench_root / "synth/alpha_stable_iso/gaussian_flow_ot.yaml")
    image_cfg = load_yaml(bench_root / "image/lvis/tedm_origin.yaml")
    hrrr_cfg = load_yaml(bench_root / "image/hrrr/ddpm_v.yaml")
    imagenet_cfg = load_yaml(bench_root / "image/imagenet_lt/dlpm_eps.yaml")

    synth_train = synth_cfg["trains"][synth_cfg["sweep"]["trains"][0]]
    image_train = image_cfg["trains"][image_cfg["sweep"]["trains"][0]]
    hrrr_train = hrrr_cfg["trains"][hrrr_cfg["sweep"]["trains"][0]]
    imagenet_train = imagenet_cfg["trains"][imagenet_cfg["sweep"]["trains"][0]]

    assert synth_train["n_epochs"] == 512
    assert image_train["n_epochs"] == 128
    assert hrrr_train["n_epochs"] == 128
    assert imagenet_train["n_epochs"] == 128
    assert synth_train["lr"] == synth_cfg["selection"]["selected_lr"]
    assert image_train["lr"] == image_cfg["selection"]["selected_lr"]
    assert hrrr_train["lr"] == hrrr_cfg["selection"]["selected_lr"]
    assert imagenet_train["lr"] == imagenet_cfg["selection"]["selected_lr"]
    assert synth_cfg["selection"]["selection_metric"] == "INNER_LOSS_VAL"
    assert image_cfg["selection"]["selection_metric"] == "INNER_LOSS_VAL"


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
    assert len(frame) == 30
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


def test_preprocessed_real_dataset_cache_roundtrip(tmp_path, monkeypatch):
    dataset_cfg = {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 20, "max_samples": 4, "seed": 0},
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

    assert result["status"] == "built"
    assert x_train.shape == (2, 3)
    assert x_val.tolist() == [[2.0, 2.0, 2.0]]
    assert x_test.tolist() == [[3.0, 3.0, 3.0]]


def test_slurm_strict_real_dataset_loading_requires_processed_cache(tmp_path, monkeypatch):
    dataset_cfg = {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 20, "max_samples": 4, "seed": 0},
        "split": {"val_size": 0.25, "test_size": 0.25, "random_state": 0, "standardize": False},
    }

    monkeypatch.delenv("FLOWBENCH_DATA", raising=False)
    monkeypatch.setenv("FLOWBENCH_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA", "1")

    with pytest.raises(FileNotFoundError, match="Missing preprocessed real dataset cache"):
        build_dataset(dataset_cfg, torch.float32, "cpu")


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
        x,
        x,
        net,
    )

    assert config["run"]["name"] == "run_name"
    assert config["run"]["seed"] == 12
    assert config["run"]["n_trial"] == 3
    assert config["run"]["trial_idx"] == 2
    assert config["run"]["resolved_dtype"] == "torch.float32"


# ---------------------------------------------------------------------------
# Cache validation: SKIP/REBUILD logging and offline correctness
# ---------------------------------------------------------------------------


def _make_cifar100_lt_cfg():
    return {
        "kind": "real",
        "name": "cifar100_lt",
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 20, "max_samples": 4, "seed": 0},
        "split": {"val_size": 0.25, "test_size": 0.25, "random_state": 0, "standardize": False},
    }


def _fake_fetch_real_4samples(name, **kwargs):
    return (
        torch.ones(2, 3),
        torch.full((1, 3), 2.0),
        torch.full((1, 3), 3.0),
        {"source": "test"},
    )


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
            "params": {"split": "train", "image_size": 32, "imbalance_factor": 20, "max_samples": 4096, "seed": 0},
            "split": {"val_size": 0.1, "test_size": 0.1, "random_state": 0, "standardize": False},
        },
        torch.float32,
    )
    cfg = {"kind": "real", "name": "cifar100_lt", "params": cfg_dict["params"], "split": cfg_dict["split"]}
    cache_path = real_data_cache.real_dataset_cache_path(cfg, torch.float32, data_root=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = real_data_cache.real_dataset_cache_key(cfg, torch.float32)
    torch.save(
        {
            "cache_version": real_data_cache.CACHE_VERSION,
            "cache_key": expected_key,
            "request": {"name": "cifar100_lt"},
            "x_train": torch.zeros(8, 3, 32, 32),
            "x_val": torch.zeros(4, 3, 32, 32),
            "x_test": torch.zeros(4, 3, 32, 32),
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
        "params": {"split": "train", "image_size": 64, "imbalance_factor": 20, "max_samples": 99999, "seed": 0},
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
            "params": {"split": "train", "image_size": 64, "imbalance_factor": 20, "max_samples": 256, "seed": 0},
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
    for dtype, dtype_str in [(torch.float32, "float32"), (torch.float64, "float64")]:
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
