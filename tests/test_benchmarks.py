from pathlib import Path

import torch

from benchmarks.evaluate import sample_generator_in_batches
from benchmarks.main import load_yaml, require_section, select_entries


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


def test_hrrr_unet_config_expands_to_one_network_variant():
    config = load_yaml(Path("benchmarks/configs/02_hrrr_unet.yaml"))
    variants = select_entries("networks", require_section(config, "networks"), ["unet"])

    assert len(variants) == 1
    params = variants[0]["config"]["params"]
    assert params["attention_resolutions"] == (4,)
    assert params["channel_mult"] == (1, 2, 4)
