#!/usr/bin/env python
"""Inspect benchmark evaluation artifacts visible to analyze-bench."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path
import re
from typing import Dict, List, Optional


DEFAULT_EXPECTED_DATASETS = [
    "alpha_stable_mixture_target",
    "alpha_stable_target",
    "cifar100_lt",
    "hrrr",
    "imagenet_lt",
    "lvis",
]
DEFAULT_LOG_ROOT = Path("logs")


def first_scalars_row(path: Path) -> Optional[Dict[str, str]]:
    try:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
            return next(csv.DictReader(handle), None)
    except Exception:
        return None


def parse_summary(path: Path) -> Dict[str, str]:
    values = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


def read_csv_rows(batch_dir: Path, patterns: List[str]) -> List[Dict[str, str]]:
    rows = []
    for pattern in patterns:
        for path in sorted(batch_dir.glob(pattern)):
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    rows.extend(csv.DictReader(handle))
            except OSError:
                continue
    return rows


def read_error_excerpt(path: Path, n_lines: int) -> str:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return " | ".join(lines[-max(1, int(n_lines)):])


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def log_tail(path: Path, n_lines: int) -> str:
    lines = [line.strip() for line in read_text(path).splitlines() if line.strip()]
    return " | ".join(lines[-max(1, int(n_lines)):])


def find_matching_logs(log_root: Path, source_batch_dir: Path, max_matches: int) -> List[Path]:
    if not log_root.is_dir():
        return []

    needles = {source_batch_dir.name, source_batch_dir.as_posix(), str(source_batch_dir.resolve())}
    matches = []
    for path in sorted(log_root.glob("*")):
        if not path.is_file() or path.suffix not in {".out", ".err"}:
            continue
        if not (path.name.startswith("heavy_tail_dynamics_bench") or path.name.startswith("heavy_tail_dynamics_run")):
            continue
        text = read_text(path)
        if any(needle and needle in text for needle in needles):
            matches.append(path)
            sibling_suffix = ".err" if path.suffix == ".out" else ".out"
            sibling = path.with_suffix(sibling_suffix)
            if sibling.is_file():
                matches.append(sibling)
        if len(matches) >= int(max_matches):
            break

    unique = []
    seen = set()
    for path in matches:
        if path in seen:
            continue
        unique.append(path)
        seen.add(path)
    return unique[: int(max_matches)]


def source_run_diagnostics(run_dirs: List[Path], n_error_lines: int) -> tuple[Counter, Counter, List[str]]:
    statuses = Counter()
    error_types = Counter()
    details = []
    for run_dir in run_dirs:
        summary = parse_summary(run_dir / "summary.txt")
        status = summary.get("status", "")
        if status:
            statuses[status] += 1
        checkpoint_exists = (run_dir / "checkpoint.pt").is_file()
        if checkpoint_exists:
            continue

        error_type = summary.get("error_type", "")
        error_message = summary.get("error_message", "")
        if error_type:
            error_types[error_type] += 1

        if error_type or error_message:
            detail = f"{run_dir.name}: {error_type or 'error'}: {error_message}".strip()
        else:
            excerpt = read_error_excerpt(run_dir / "error.txt", n_error_lines)
            detail = f"{run_dir.name}: {excerpt}" if excerpt else f"{run_dir.name}: no checkpoint and no run error file"
        details.append(detail)
    return statuses, error_types, details


def dataset_from_run_name(run_name: str) -> str:
    match = re.match(r"^[0-9]+_+(.+?)__", run_name)
    return match.group(1) if match else ""


def dataset_from_batch_name(batch_name: str) -> str:
    name = batch_name.removesuffix("_evaluate")
    match = re.search(r"_bench_(?:image|synth)__(.+?)__", name)
    return match.group(1) if match else ""


def discover_like_plotting(batch_dir: Path) -> str:
    for scalars_path in sorted(batch_dir.glob("[0-9][0-9][0-9]*/scalars.csv.gz")):
        row = first_scalars_row(scalars_path)
        if row:
            return row.get("dataset_preset") or row.get("dataset_name") or ""
    return ""


def int_value(values: Dict[str, str], key: str) -> int:
    try:
        return int(values.get(key, 0))
    except ValueError:
        return 0


def inspect_artifacts(
    artifact_root: Path,
    expected_datasets: List[str],
    n_error_lines: int,
    log_root: Path,
    n_log_lines: int,
    max_log_matches: int,
) -> int:
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Missing artifact root: {artifact_root}")

    discovered: Dict[str, List[Path]] = defaultdict(list)
    batch_rows = []
    for batch_dir in sorted(path for path in artifact_root.glob("*_evaluate") if path.is_dir()):
        if "pilot" in batch_dir.name.lower():
            continue

        discovered_dataset = discover_like_plotting(batch_dir)
        if discovered_dataset:
            discovered[discovered_dataset].append(batch_dir)

        run_dirs = sorted(path for path in batch_dir.iterdir() if path.is_dir() and path.name[:3].isdigit())
        scalar_paths = sorted(batch_dir.glob("[0-9][0-9][0-9]*/scalars.csv.gz"))
        source_batch_dir = batch_dir.with_name(batch_dir.name.removesuffix("_evaluate"))
        source_run_dirs = []
        source_checkpoints = []
        source_statuses = Counter()
        source_error_types = Counter()
        source_error_details = []
        source_log_details = []
        if source_batch_dir.is_dir():
            source_run_dirs = sorted(path for path in source_batch_dir.iterdir() if path.is_dir() and path.name[:3].isdigit())
            source_checkpoints = sorted(source_batch_dir.glob("[0-9][0-9][0-9]*/checkpoint.pt"))
            source_manifest_rows = read_csv_rows(source_batch_dir, ["manifest.csv", "manifest_shard_*.csv"])
            source_summaries = [parse_summary(path) for path in sorted(source_batch_dir.glob("summary*.txt"))]
            run_statuses, run_error_types, source_error_details = source_run_diagnostics(source_run_dirs, n_error_lines)
            source_statuses = Counter(row.get("status", "") for row in source_manifest_rows)
            source_error_types = Counter(row.get("error_type", "") for row in source_manifest_rows if row.get("status") == "failed")
            if not source_statuses:
                source_statuses.update(run_statuses)
            if not source_error_types:
                source_error_types.update(run_error_types)
            if not source_statuses:
                source_statuses["failed"] = sum(int_value(summary, "n_failed") for summary in source_summaries)
                source_statuses["skipped"] = sum(int_value(summary, "n_skipped") for summary in source_summaries)
            if source_run_dirs and not source_checkpoints:
                for path in find_matching_logs(log_root, source_batch_dir, max_log_matches):
                    tail = log_tail(path, n_log_lines)
                    if tail:
                        source_log_details.append(f"{path}: {tail}")
        summaries = [parse_summary(path) for path in sorted(batch_dir.glob("summary_eval*.txt"))]
        manifests = read_csv_rows(batch_dir, ["manifest_eval*.csv"])
        statuses = Counter(row.get("status", "") for row in manifests)
        error_types = Counter(row.get("error_type", "") for row in manifests if row.get("status") == "failed")

        dataset_guess = discovered_dataset
        if not dataset_guess:
            dataset_guess = next((dataset_from_run_name(path.name) for path in run_dirs if dataset_from_run_name(path.name)), "")
        if not dataset_guess:
            dataset_guess = dataset_from_batch_name(batch_dir.name)
        if not dataset_guess:
            dataset_guess = "?"
        batch_rows.append(
            {
                "dataset": dataset_guess,
                "batch": batch_dir.name,
                "visible": "yes" if discovered_dataset else "no",
                "runs": len(run_dirs),
                "scalars": len(scalar_paths),
                "source_runs": len(source_run_dirs),
                "source_ckpts": len(source_checkpoints),
                "source_failed": source_statuses.get("failed", 0),
                "source_errors": ", ".join(f"{name}:{count}" for name, count in source_error_types.most_common(3) if name),
                "source_details": source_error_details[:3],
                "source_logs": source_log_details[:3],
                "done": sum(int_value(summary, "n_done") for summary in summaries) or statuses.get("ok", 0),
                "skipped": sum(int_value(summary, "n_skipped") for summary in summaries) or statuses.get("skipped", 0),
                "failed": sum(int_value(summary, "n_failed") for summary in summaries) or statuses.get("failed", 0),
                "errors": ", ".join(f"{name}:{count}" for name, count in error_types.most_common(3) if name),
            }
        )

    print("[plot-discovery] datasets visible to make analyze-bench")
    if discovered:
        for dataset, batches in sorted(discovered.items()):
            n_scalars = sum(1 for batch in batches for _ in batch.glob("[0-9][0-9][0-9]*/scalars.csv.gz"))
            print(f"  {dataset:<32} batches={len(batches):<3} scalars={n_scalars}")
    else:
        print("  none")

    missing = [dataset for dataset in expected_datasets if dataset not in discovered]
    if missing:
        print("\n[missing expected datasets]")
        for dataset in missing:
            candidates = [row for row in batch_rows if row["dataset"] == dataset or dataset in row["batch"]]
            if not candidates:
                print(f"  {dataset:<32} no non-pilot *_evaluate batch found")
                continue
            for row in candidates:
                print(
                    f"  {dataset:<32} visible={row['visible']:<3} scalars={row['scalars']:<3} "
                    f"src_ckpts={row['source_ckpts']:<3} src_fail={row['source_failed']:<3} "
                    f"done={row['done']:<3} skipped={row['skipped']:<3} failed={row['failed']:<3} "
                    f"batch={row['batch']} errors={row['source_errors'] or row['errors'] or '-'}"
                )
    else:
        print("\n[missing expected datasets] none")

    print("\n[all non-pilot evaluation batches]")
    header = f"{'dataset':<32} {'vis':<3} {'runs':>4} {'scalars':>7} {'src':>4} {'ckpt':>4} {'sfail':>5} {'done':>5} {'skip':>5} {'fail':>5} batch"
    print(header)
    print("-" * len(header))
    for row in batch_rows:
        print(
            f"{row['dataset']:<32} {row['visible']:<3} {row['runs']:>4} {row['scalars']:>7} "
            f"{row['source_runs']:>4} {row['source_ckpts']:>4} {row['source_failed']:>5} "
            f"{row['done']:>5} {row['skipped']:>5} {row['failed']:>5} {row['batch']}"
        )
        if row["source_errors"]:
            print(f"{'':<65} source errors: {row['source_errors']}")
        for detail in row["source_details"]:
            print(f"{'':<65} source detail: {detail}")
        for detail in row["source_logs"]:
            print(f"{'':<65} source log: {detail}")
        if row["errors"]:
            print(f"{'':<65} errors: {row['errors']}")
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmarks/artifacts"))
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--expected-datasets", nargs="*", default=DEFAULT_EXPECTED_DATASETS)
    parser.add_argument("--error-lines", type=int, default=4, help="Number of trailing error.txt lines to print for source runs without checkpoints.")
    parser.add_argument("--log-lines", type=int, default=8, help="Number of trailing Slurm log lines to print for source batches without checkpoints.")
    parser.add_argument("--max-log-matches", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(
        inspect_artifacts(
            args.artifact_root,
            list(args.expected_datasets),
            args.error_lines,
            args.log_root,
            args.log_lines,
            args.max_log_matches,
        )
    )


if __name__ == "__main__":
    main()
