"""Shared benchmark helpers."""

import argparse
import copy
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML config file."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to a mapping.")
    return data


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one required config section."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def _grid_items(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict) and set(value) == {"literal"}:
        literal = copy.deepcopy(value["literal"])
        if isinstance(literal, list):
            literal = tuple(literal)
        return [(prefix, literal)]
    if not isinstance(value, dict):
        return [(prefix, value)]
    items: list[tuple[str, Any]] = []
    for key, child in value.items():
        items.extend(_grid_items(f"{prefix}.{key}" if prefix else str(key), child))
    return items


def _assign_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def select_entries(section_name: str, registry: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """Select and expand named presets for one registry section."""
    variants = []
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown {section_name} entry {name!r}. Available: {sorted(registry)}")
        entry_cfg = registry[name]
        if not isinstance(entry_cfg, dict):
            raise ValueError(f"{section_name}.{name} must be a mapping.")
        static_cfg: dict[str, Any] = {}
        varying_items: list[tuple[str, list[Any]]] = []
        for key, value in _grid_items("", copy.deepcopy(entry_cfg)):
            if isinstance(value, list):
                if not value:
                    raise ValueError(f"Entry {name!r} has an empty grid at {key!r}.")
                varying_items.append((key, value))
            else:
                _assign_path(static_cfg, key, value)

        if not varying_items:
            variants.append({"entry_name": name, "config": static_cfg, "variant_name": name, "variant_params": {}})
            continue

        keys = [key for key, _ in varying_items]
        for combo in itertools.product(*(choices for _, choices in varying_items)):
            variant_cfg = copy.deepcopy(static_cfg)
            variant_params = dict(zip(keys, combo, strict=True))
            for key, value in variant_params.items():
                _assign_path(variant_cfg, key, value)
            suffix = "__".join(
                f"{key.split('.')[-1]}-{str(value).strip().replace('/', '-').replace(' ', '_').replace('.', 'p')}"
                for key, value in variant_params.items()
            )
            variants.append(
                {
                    "entry_name": name,
                    "config": variant_cfg,
                    "variant_name": f"{name}__{suffix}",
                    "variant_params": variant_params,
                }
            )
    return variants


def _path_suffix_from_marker(path: Path | str, marker: str) -> str:
    parts = Path(str(path)).parts
    if marker in parts:
        return "/".join(parts[parts.index(marker):])
    return Path(str(path)).as_posix().lstrip("./")


def latest_config_batch_dir(root: Path, config_path: Path) -> Path:
    """Return the latest non-evaluation batch directory for one benchmark config."""
    root = root.expanduser()
    config_path = config_path.expanduser()
    run_name = str(load_yaml(config_path)["run"]["name"])
    candidates = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.endswith("_evaluate"):
            continue
        if path.name.endswith(f"_{run_name}"):
            candidates.append(path)
            continue
        summary_config_matches = False
        summary_paths = [path / "summary.txt", *sorted(path.glob("summary_shard_*.txt"))]
        for summary_path in summary_paths:
            if not summary_path.is_file():
                continue
            try:
                lines = summary_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.startswith("config:"):
                    continue
                recorded_config = line.split(":", 1)[1].strip()
                if not recorded_config:
                    continue
                left_path = Path(recorded_config).expanduser()
                right_path = config_path.expanduser()
                candidates_left = {recorded_config, left_path.as_posix()}
                candidates_right = {str(config_path), right_path.as_posix()}
                if left_path.exists():
                    candidates_left.add(str(left_path.resolve()))
                if right_path.exists():
                    candidates_right.add(str(right_path.resolve()))
                same_config = bool(candidates_left & candidates_right)
                if not same_config:
                    same_config = _path_suffix_from_marker(left_path, "benchmarks") == _path_suffix_from_marker(right_path, "benchmarks")
                if same_config:
                    summary_config_matches = True
                    break
            if summary_config_matches:
                break

        saved_run_name_matches = False
        for run_config_path in sorted(path.glob("[0-9][0-9][0-9]*/config.yaml")):
            try:
                run_cfg = load_yaml(run_config_path).get("run", {})
            except (OSError, ValueError, yaml.YAMLError):
                continue
            if isinstance(run_cfg, dict) and str(run_cfg.get("name", "")) == run_name:
                saved_run_name_matches = True
                break

        if summary_config_matches or saved_run_name_matches:
            candidates.append(path)
    matches = sorted(set(candidates), key=lambda path: (path.stat().st_mtime, path.name))
    if not matches:
        raise FileNotFoundError(f"No batch directory for config {config_path} under {root}")
    return matches[-1]


def make_batch_dir(run_cfg: dict[str, Any], save_cfg: dict[str, Any]) -> Path:
    """Create the parent output directory for one config sweep."""
    root = Path(save_cfg.get("root_dir", "runs")).expanduser()
    name = str(run_cfg.get("name", "run")).strip() or "run"
    batch_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    latest = subparsers.add_parser("latest-batch", help="Print the latest directory matching one glob pattern.")
    latest.add_argument("--root", type=Path, required=True)
    latest.add_argument("--pattern", required=True)

    latest_config = subparsers.add_parser("latest-config-batch", help="Print the latest batch directory for one config.")
    latest_config.add_argument("--root", type=Path, required=True)
    latest_config.add_argument("--config", type=Path, required=True)

    count_runs = subparsers.add_parser("count-runs", help="Print the expanded run count for one config.")
    count_runs.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "latest-batch":
        matches = sorted(path for path in Path(args.root).expanduser().glob(args.pattern) if path.is_dir())
        if not matches:
            raise FileNotFoundError(f"No directory matching {args.pattern!r} under {args.root}")
        print(matches[-1].resolve())
    elif args.command == "latest-config-batch":
        print(latest_config_batch_dir(args.root, args.config).resolve())
    elif args.command == "count-runs":
        config = load_yaml(args.config)
        sweep = require_section(config, "sweep")
        n_runs = int(config["run"]["n_trial"])
        for section in ["datasets", "networks", "models", "trains"]:
            variants = select_entries(section, require_section(config, section), list(sweep[section]))
            n_runs *= len(variants)
        print(n_runs)
    else:
        raise ValueError(f"Unknown command: {args.command}")
