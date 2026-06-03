"""Dataset staging helpers for datakit real datasets."""

import argparse
import os
from importlib.resources import files
from pathlib import Path
import shutil
from typing import Callable
from urllib.request import urlretrieve


_CIFAR100_SOURCE = Path("/lustre/fsmisc/dataset/cifar-100-python")
_IMAGENET_SOURCE = Path("/lustre/fswork/dataset/imagenet")
_WILDFIRES_URL = "https://aaronclauset.github.io/powerlaws/data/fires.txt"


def _default_root(dataset: str, subdir: str) -> Path:
    flowbench_data = os.getenv("FLOWBENCH_DATA")
    if not flowbench_data:
        raise RuntimeError(f"FLOWBENCH_DATA must be set, or pass --root <{dataset}_root>.")
    return Path(flowbench_data).expanduser() / "raw" / subdir


def _symlink_or_keep(source: Path, link: Path, *, label: str) -> None:
    if link.is_symlink() or link.exists():
        print(f"[datakit] {label} already present at {link}", flush=True)
        return
    if not source.is_dir():
        raise RuntimeError(f"Source directory not found: {source}")
    link.symlink_to(source)
    print(f"[datakit] symlinked {source} -> {link}", flush=True)


def prepare_cifar100_lt(root: str | Path | None = None, source: str | Path | None = None) -> None:
    root_path = Path(root).expanduser() if root is not None else _default_root("cifar100_lt", "cifar100_lt")
    source_path = Path(source).expanduser() if source is not None else _CIFAR100_SOURCE
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"[datakit] cifar100_lt root: {root_path}", flush=True)

    link = root_path / "cifar-100-python"
    _symlink_or_keep(source_path, link, label="cifar-100-python")
    for name in ("train", "test", "meta"):
        if not (link / name).is_file():
            raise RuntimeError(f"Expected CIFAR-100 file missing: {link / name}")
    print("[datakit] cifar100_lt done", flush=True)


def prepare_imagenet_lt(root: str | Path | None = None, source: str | Path | None = None) -> None:
    root_path = Path(root).expanduser() if root is not None else _default_root("imagenet_lt", "imagenet_lt")
    source_path = Path(source).expanduser() if source is not None else _IMAGENET_SOURCE
    annotation_dir = root_path / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    print(f"[datakit] imagenet_lt root: {root_path}", flush=True)

    resource_dir = files("datakit").joinpath("resources/imagenet_lt/annotations")
    for name in ("ImageNet_LT_train.txt", "ImageNet_LT_val.txt", "ImageNet_LT_test.txt"):
        source_file = resource_dir.joinpath(name)
        target_file = annotation_dir / name
        if target_file.is_file():
            print(f"[datakit] annotation already present: {target_file}", flush=True)
            continue
        with source_file.open("rb") as src, target_file.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"[datakit] copied {source_file} -> {target_file}", flush=True)

    _symlink_or_keep(source_path, root_path / "imagenet", label="imagenet tree")
    print("[datakit] imagenet_lt done", flush=True)


def prepare_wildfires(root: str | Path | None = None) -> None:
    root_path = Path(root).expanduser() if root is not None else _default_root("wildfires", "powerlaws")
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / "fires.txt"
    if target.is_file() and target.stat().st_size > 0:
        print(f"[datakit] wildfires already present: {target}", flush=True)
        return

    tmp_target = target.with_suffix(".txt.tmp")
    print(f"[datakit] downloading {_WILDFIRES_URL} -> {target}", flush=True)
    try:
        urlretrieve(_WILDFIRES_URL, str(tmp_target))
        tmp_target.replace(target)
    finally:
        if tmp_target.exists():
            tmp_target.unlink()
    print("[datakit] wildfires done", flush=True)


def _add_common_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datakit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="stage raw files for a real dataset")
    init_subparsers = init_parser.add_subparsers(dest="dataset", required=True)

    commands: dict[str, Callable[..., None]] = {
        "cifar100_lt": prepare_cifar100_lt,
        "imagenet_lt": prepare_imagenet_lt,
        "wildfires": prepare_wildfires,
    }
    for name, command in commands.items():
        dataset_parser = init_subparsers.add_parser(name)
        if name == "wildfires":
            dataset_parser.add_argument("--root", type=Path, default=None)
        else:
            _add_common_source_args(dataset_parser)
        dataset_parser.set_defaults(func=command)

    args = parser.parse_args(argv)
    kwargs = vars(args)
    command = kwargs.pop("func")
    kwargs.pop("command", None)
    kwargs.pop("dataset", None)
    try:
        command(**kwargs)
    except RuntimeError as exc:
        parser.exit(1, f"[datakit] {exc}\n")
    return 0
