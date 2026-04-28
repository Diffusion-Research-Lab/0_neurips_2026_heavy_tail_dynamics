from pathlib import Path

import torch

import benchmarks.evaluate as evaluate
from benchmarks.evaluate import compute_test_metrics, sample_generator_in_batches
from benchmarks.main import _resolved_config, load_yaml, require_section, select_entries


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


def test_compute_test_metrics_keeps_other_metrics_when_one_fails(monkeypatch):
    def raise_fid(*args, **kwargs):
        raise RuntimeError("fid boom")

    monkeypatch.setattr(evaluate, "fid", raise_fid)
    x_ref = torch.arange(256, dtype=torch.float32).reshape(128, 2) + 1.0
    x_gen = x_ref + 0.1

    values, warnings = compute_test_metrics(
        x_ref,
        x_gen,
        feature_dim=2,
        max_fid_dim=4,
        max_mmd_dim=4,
        max_mmd_samples=8,
    )

    assert values["FID"] != values["FID"]
    assert any("FID_failed: RuntimeError: fid boom" == warning for warning in warnings)
    for metric_name in ["MMD_RBF", "SLICED_WASSERSTEIN", "TAIL_COVERAGE_ERROR", "MSSLE"]:
        assert torch.isfinite(torch.tensor(values[metric_name]))


def test_image_bench_unet_config_expands_to_one_network_variant():
    config = load_yaml(Path("benchmarks/configs/04_image_bench.yaml"))
    variants = select_entries("networks", require_section(config, "networks"), ["unet"])

    assert len(variants) == 1
    params = variants[0]["config"]["params"]
    assert params["attention_resolutions"] == (4,)
    assert params["channel_mult"] == (1, 2, 4)


def test_pilot_config_run_counts_match_makefile_arrays():
    expected_counts = {
        "01_alphastable_pilot.yaml": 270,
        "02_alphastable_bench.yaml": 100,
        "03_image_pilot.yaml": 40,
        "04_image_bench.yaml": 10,
    }

    for filename, expected_count in expected_counts.items():
        config = load_yaml(Path("benchmarks/configs") / filename)
        sweep = require_section(config, "sweep")
        n_trial = int(config["run"]["n_trial"])
        n_runs = n_trial
        for section in ["datasets", "networks", "models", "trains"]:
            variants = select_entries(section, require_section(config, section), list(sweep[section]))
            n_runs *= len(variants)
        assert n_runs == expected_count


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
