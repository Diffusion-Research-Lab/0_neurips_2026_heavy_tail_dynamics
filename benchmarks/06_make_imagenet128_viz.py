#!/usr/bin/env python
"""Create ImageNet-LT-128 visualization benchmark configs from pilot-selected bench configs."""

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import yaml  # noqa
from benchmarks.utils import load_yaml  # noqa


MODEL_ORDER = [
    "gaussian_flow_linear_euler",
    "gaussian_flow_linear_heun",
    "ddpm_v_ddpm",
    "ddpm_v_ddim",
    "dlpm_eps_a17",
    "dlpm_eps_a19",
    "tedm_origin_nu21",
    "tedm_origin_nu30",
]


def _imagenet128_dataset(source_dataset: dict[str, Any]) -> dict[str, Any]:
    dataset = copy.deepcopy(source_dataset)
    dataset["name"] = "imagenet_lt"
    params = copy.deepcopy(dataset.get("params", {}))
    params["split"] = params.get("split", "train")
    params["image_size"] = 128
    params.pop("max_samples", None)
    params.pop("n_samples", None)
    dataset["params"] = params
    return dataset


def _imagenet128_network(source_network: dict[str, Any]) -> dict[str, Any]:
    network = copy.deepcopy(source_network)
    params = copy.deepcopy(network.get("params", {}))
    for key in ("sample_size", "in_channels", "out_channels"):
        params.pop(key, None)
    network["params"] = params
    return network


def build_viz_config(source_config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return one ImageNet-LT-128 config preserving one selected model/training pair."""
    if model_name not in source_config.get("models", {}):
        raise KeyError(f"Source config does not define model {model_name!r}.")
    if len(source_config.get("sweep", {}).get("networks", [])) != 1:
        raise ValueError("Source config must select exactly one network.")
    if len(source_config.get("sweep", {}).get("trains", [])) != 1:
        raise ValueError("Source config must select exactly one train preset.")

    source_dataset_name = source_config["sweep"]["datasets"][0]
    network_name = source_config["sweep"]["networks"][0]
    train_name = source_config["sweep"]["trains"][0]
    dataset_name = "imagenet_lt_128"

    selection = copy.deepcopy(source_config.get("selection", {}))
    selection.update(
        {
            "dataset_slug": dataset_name,
            "source_dataset_slug": source_config.get("selection", {}).get("dataset_slug", "imagenet_lt"),
            "source_config": source_config.get("run", {}).get("resolved_config_path"),
            "note": "ImageNet-LT-128 visualization config generated from pilot-selected ImageNet-LT benchmark settings.",
        }
    )

    return {
        "run": {
            **copy.deepcopy(source_config["run"]),
            "name": f"viz_image__imagenet_lt_128__{model_name}",
        },
        "selection": selection,
        "sweep": {
            "datasets": [dataset_name],
            "networks": [network_name],
            "models": [model_name],
            "trains": [train_name],
        },
        "datasets": {
            dataset_name: _imagenet128_dataset(source_config["datasets"][source_dataset_name]),
        },
        "networks": {
            network_name: _imagenet128_network(source_config["networks"][network_name]),
        },
        "models": {
            model_name: copy.deepcopy(source_config["models"][model_name]),
        },
        "trains": {
            train_name: copy.deepcopy(source_config["trains"][train_name]),
        },
        "save": copy.deepcopy(source_config["save"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-config-root", type=Path, default=PROJECT_ROOT / "benchmarks" / "configs" / "bench")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "benchmarks" / "configs" / "viz" / "imagenet_lt_128")
    parser.add_argument("--dataset-slug", default="imagenet_lt")
    args = parser.parse_args()

    source_root = args.bench_config_root / "image" / args.dataset_slug
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing {source_root}. Run `make analyze-pilot` first.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    written = []
    for model_name in MODEL_ORDER:
        source_path = source_root / f"{model_name}.yaml"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing selected bench config: {source_path}")
        config = build_viz_config(load_yaml(source_path), model_name)
        output_path = args.output_root / f"{model_name}.yaml"
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        written.append(output_path)

    print("Wrote ImageNet-LT-128 visualization configs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
