#!/usr/bin/env python
"""Ensure the raw HRRR APCP tensor has enough hourly samples."""

import argparse
import collections
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


START = pd.Timestamp("2014-07-30 00:00:00")
PATCH_SIZE = 100
CENTER_LAT = 39.0
CENTER_LON = -98.0
SEARCH = r":APCP:surface:0-1 hour acc fcst:"
DEFAULT_MIN_COVERAGE = 0.90
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_FLUSH_EVERY = 25
DEFAULT_MAX_FAILURE_LOGS = 8
SHARD_DIR_NAME = "hrrr_apcp_100x100_shards"


def require_work() -> Path:
    work_value = os.environ.get("WORK")
    if work_value is None:
        raise RuntimeError("$WORK is not set")
    work = Path(work_value).expanduser()
    if not work.is_dir():
        raise FileNotFoundError(f"$WORK does not exist or is not a directory: {work}")
    return work


def default_output_path() -> Path:
    return require_work() / "hrrr_data" / "hrrr_apcp_100x100.pt"


def parse_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def default_end() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h") - pd.Timedelta(hours=2)


def n_hourly_samples(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end < start:
        raise ValueError(f"end must be >= start, got start={start} end={end}")
    return int(len(pd.date_range(start, end, freq="h")))


def required_samples(start: pd.Timestamp, end: pd.Timestamp, min_samples: int | None, min_coverage: float) -> int:
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1].")
    coverage_count = math.ceil(n_hourly_samples(start, end) * min_coverage)
    return max(coverage_count, int(min_samples or 0))


def count_hrrr_samples(path: Path) -> int:
    if not path.is_file():
        return 0
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    tensor = hrrr_tensor_from_payload(payload, path)
    return int(tensor.shape[0])


def normalize_dataset(ds: Any) -> Any:
    if isinstance(ds, list):
        if len(ds) != 1:
            raise RuntimeError(f"Expected exactly one dataset, received {len(ds)}")
        ds = ds[0]
    return ds


def extract_field(ds: Any) -> np.ndarray:
    for name in ds.data_vars:
        array = ds[name].squeeze(drop=True)
        if set(array.dims) == {"y", "x"}:
            return np.asarray(array.transpose("y", "x").values, dtype=np.float32)
    raise RuntimeError(f"No two-dimensional y/x field found. Variables: {list(ds.data_vars)}")


def open_apcp(date: pd.Timestamp, grib_dir: str) -> Any:
    from herbie import Herbie

    hrrr = Herbie(date, model="hrrr", product="sfc", fxx=1, save_dir=grib_dir, verbose=False)
    return normalize_dataset(hrrr.xarray(SEARCH, remove_grib=True))


def read_reference_field(date: pd.Timestamp, grib_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = open_apcp(date, str(grib_dir))
    field = extract_field(ds)
    latitude = np.asarray(ds.latitude.values, dtype=np.float32)
    longitude = np.asarray(ds.longitude.values, dtype=np.float32)
    ds.close()
    return field, latitude, longitude


def fetch_one(task: tuple[int, str, int, int, str]) -> tuple[int, np.ndarray | None, str]:
    index, date_string, y0, x0, grib_dir = task
    date = pd.Timestamp(date_string)
    for attempt in range(3):
        try:
            ds = open_apcp(date, grib_dir)
            field = extract_field(ds)
            ds.close()
            patch = field[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                raise RuntimeError(f"Unexpected patch shape {patch.shape}")
            return index, np.maximum(patch, 0.0).astype(np.float32, copy=False), ""
        except Exception as exc:
            if attempt == 2:
                return index, None, repr(exc)
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def compact_fetch_error(error: str) -> str:
    if "No index file was found" in error:
        return "missing Herbie index"
    if "ProxyError" in error:
        return "proxy/network error"
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    return (lines[0] if lines else error)[:240]


def pending_indices(status: np.ndarray, indices: np.ndarray, *, retry_failed: bool) -> list[int]:
    if retry_failed:
        return [int(index) for index in indices if status[index] != 1]
    return [int(index) for index in indices if status[index] == 0]


def find_patch(end: pd.Timestamp, grib_dir: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    last_error = None
    for date in pd.date_range(end=end, periods=48, freq="h")[::-1]:
        try:
            _, latitude, longitude = read_reference_field(date, grib_dir)
            target_lon = CENTER_LON % 360.0
            delta_lon = (longitude - target_lon + 180.0) % 360.0 - 180.0
            distance_squared = (latitude - CENTER_LAT) ** 2 + (np.cos(np.deg2rad(CENTER_LAT)) * delta_lon) ** 2
            center_y, center_x = np.unravel_index(np.nanargmin(distance_squared), distance_squared.shape)
            y0 = int(np.clip(center_y - PATCH_SIZE // 2, 0, latitude.shape[0] - PATCH_SIZE))
            x0 = int(np.clip(center_x - PATCH_SIZE // 2, 0, latitude.shape[1] - PATCH_SIZE))
            latitude_patch = latitude[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            longitude_patch = longitude[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            longitude_patch = (longitude_patch + 180.0) % 360.0 - 180.0
            return y0, x0, latitude_patch, longitude_patch
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Could not find a recent HRRR F01 APCP file") from last_error


def write_status(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hrrr_tensor_from_payload(payload: Any, path: Path) -> Any:
    import torch

    tensor = payload
    if isinstance(payload, dict):
        tensor = payload.get("frames", payload.get("apcp_mm"))
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"HRRR file does not contain a tensor, 'frames', or 'apcp_mm': {path}")
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (1, PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(
            f"Unexpected HRRR tensor shape in {path}: {tuple(tensor.shape)}. "
            f"Expected (N, 1, {PATCH_SIZE}, {PATCH_SIZE})."
        )
    return tensor


def save_hrrr_payload(
    path: Path,
    frames: Any,
    init_time_unix_s: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    complete: bool,
    min_required: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_tensor = frames if isinstance(frames, torch.Tensor) else torch.as_tensor(frames)
    frame_tensor = frame_tensor.detach().cpu().contiguous()
    init_time = torch.as_tensor(init_time_unix_s.copy(), dtype=torch.int64)
    payload = {
        "frames": frame_tensor,
        "apcp_mm": frame_tensor,
        "init_time_unix_s": init_time,
        "valid_time_unix_s": init_time + 3600,
        "latitude": torch.as_tensor(latitude.astype(np.float32, copy=False)),
        "longitude": torch.as_tensor(longitude.astype(np.float32, copy=False)),
        "forecast_hour": 1,
        "units": "mm",
        "variable": "APCP",
        "center_latitude": CENTER_LAT,
        "center_longitude": CENTER_LON,
        "complete": bool(complete),
        "min_required": int(min_required),
        "n_samples": int(frame_tensor.shape[0]),
    }
    staged_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, staged_path)
    os.replace(staged_path, path)


def write_checkpoint_metadata(
    shard_dir: Path,
    config: dict[str, Any],
    latitude_patch: np.ndarray,
    longitude_patch: np.ndarray,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "latitude.npy", latitude_patch.astype(np.float32, copy=False))
    np.save(shard_dir / "longitude.npy", longitude_patch.astype(np.float32, copy=False))
    staged_path = shard_dir / f".metadata.{os.getpid()}.{time.time_ns()}.json"
    staged_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staged_path, shard_dir / "metadata.json")


def load_checkpoint_config(temporary_dir: Path, shard_dir: Path) -> dict[str, Any] | None:
    for path in (temporary_dir / "config.json", shard_dir / "metadata.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def open_or_create_memmaps(
    raw_path: Path,
    status_path: Path,
    expected_shape: tuple[int, int, int, int],
) -> tuple[np.memmap, np.memmap]:
    status_shape = (expected_shape[0],)
    if raw_path.exists() != status_path.exists():
        raise RuntimeError(f"Found only one HRRR temporary file among {raw_path} and {status_path}.")
    if not raw_path.exists():
        raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=expected_shape)
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.uint8, shape=status_shape)
        status[:] = 0
        status.flush()
        return raw, status

    raw = np.lib.format.open_memmap(raw_path, mode="r+")
    status = np.lib.format.open_memmap(status_path, mode="r+")
    if raw.shape == expected_shape and status.shape == status_shape:
        return raw, status
    if raw.shape[1:] != expected_shape[1:] or status.shape != (raw.shape[0],) or raw.shape[0] > expected_shape[0]:
        raise RuntimeError(
            f"Temporary files are incompatible with the saved HRRR configuration: raw={raw.shape}, status={status.shape}, expected={expected_shape}."
        )

    old_n = raw.shape[0]
    new_raw_path = raw_path.with_name(f".{raw_path.stem}.extended.npy")
    new_status_path = status_path.with_name(f".{status_path.stem}.extended.npy")
    new_raw = np.lib.format.open_memmap(new_raw_path, mode="w+", dtype=np.float32, shape=expected_shape)
    new_status = np.lib.format.open_memmap(new_status_path, mode="w+", dtype=np.uint8, shape=status_shape)
    new_status[:] = 0
    for offset in range(0, old_n, 512):
        stop = min(offset + 512, old_n)
        new_raw[offset:stop] = raw[offset:stop]
    new_status[:old_n] = status[:old_n]
    new_raw.flush()
    new_status.flush()
    del raw, status, new_raw, new_status
    os.replace(new_raw_path, raw_path)
    os.replace(new_status_path, status_path)
    return np.lib.format.open_memmap(raw_path, mode="r+"), np.lib.format.open_memmap(status_path, mode="r+")


def restore_payload_into_memmaps(
    payload_path: Path,
    dates: pd.DatetimeIndex,
    raw: np.memmap,
    status: np.memmap,
) -> int:
    if not payload_path.is_file():
        return 0
    import torch

    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    tensor = hrrr_tensor_from_payload(payload, payload_path).to(dtype=torch.float32)
    if not isinstance(payload, dict) or "init_time_unix_s" not in payload:
        return 0
    init_times = torch.as_tensor(payload["init_time_unix_s"], dtype=torch.int64).cpu().numpy()
    date_to_index = {int(value): index for index, value in enumerate(dates.asi8 // 1_000_000_000)}
    restored = 0
    for frame_index, init_time in enumerate(init_times):
        index = date_to_index.get(int(init_time))
        if index is None or status[index] == 1:
            continue
        raw[index] = tensor[frame_index].cpu().numpy()
        status[index] = 1
        restored += 1
    return restored


def restore_checkpoints_into_memmaps(
    output_path: Path,
    shard_dir: Path,
    dates: pd.DatetimeIndex,
    raw: np.memmap,
    status: np.memmap,
) -> int:
    restored = restore_payload_into_memmaps(output_path, dates, raw, status)
    if shard_dir.is_dir():
        for shard_path in sorted(shard_dir.glob("*.pt")):
            restored += restore_payload_into_memmaps(shard_path, dates, raw, status)
    if restored:
        raw.flush()
        status.flush()
    return restored


def write_month_shard(
    shard_dir: Path,
    month: pd.Period,
    raw: np.memmap,
    status: np.memmap,
    dates: pd.DatetimeIndex,
    month_indices: np.ndarray,
    latitude_patch: np.ndarray,
    longitude_patch: np.ndarray,
    min_required: int,
) -> int:
    successful_indices = month_indices[np.asarray(status[month_indices]) == 1]
    if len(successful_indices) == 0:
        return 0
    init_time = dates.asi8[successful_indices] // 1_000_000_000
    frames = np.asarray(raw[successful_indices], dtype=np.float32)
    save_hrrr_payload(
        shard_dir / f"{month}.pt",
        frames,
        init_time,
        latitude_patch,
        longitude_patch,
        complete=False,
        min_required=min_required,
    )
    return int(len(successful_indices))


def build_hrrr_dataset(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
    n_workers: int,
    min_required: int,
    keep_temp: bool,
    flush_every: int,
    retry_failed: bool,
    max_failure_logs: int,
) -> int:
    import torch

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / ".hrrr_apcp_100x100_tmp"
    shard_dir = output_dir / SHARD_DIR_NAME
    grib_dir = temporary_dir / "grib"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    grib_dir.mkdir(parents=True, exist_ok=True)

    config_path = temporary_dir / "config.json"
    latitude_path = temporary_dir / "latitude.npy"
    longitude_path = temporary_dir / "longitude.npy"
    requested_end = end
    config = load_checkpoint_config(temporary_dir, shard_dir)
    if config is not None:
        start = parse_timestamp(config["start"])
        end = max(parse_timestamp(config["end"]), requested_end)
        y0 = int(config["y0"])
        x0 = int(config["x0"])
        if latitude_path.is_file() and longitude_path.is_file():
            latitude_patch = np.load(latitude_path)
            longitude_patch = np.load(longitude_path)
        else:
            latitude_patch = np.load(shard_dir / "latitude.npy")
            longitude_patch = np.load(shard_dir / "longitude.npy")
        config["end"] = end.isoformat()
    else:
        print(f"[hrrr] locating 100x100 crop from recent HRRR files ending at {end}", flush=True)
        y0, x0, latitude_patch, longitude_patch = find_patch(end, grib_dir)
        config = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "y0": y0,
            "x0": x0,
            "center_lat": CENTER_LAT,
            "center_lon": CENTER_LON,
            "patch_size": PATCH_SIZE,
        }
    np.save(latitude_path, latitude_patch)
    np.save(longitude_path, longitude_patch)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_checkpoint_metadata(shard_dir, config, latitude_patch, longitude_patch)

    dates = pd.date_range(start, end, freq="h")
    n_possible = len(dates)
    raw_path = temporary_dir / "raw.npy"
    status_path = temporary_dir / "status.npy"
    expected_shape = (n_possible, 1, PATCH_SIZE, PATCH_SIZE)
    raw, status = open_or_create_memmaps(raw_path, status_path, expected_shape)
    restored = restore_checkpoints_into_memmaps(output_path, shard_dir, dates, raw, status)
    if restored:
        print(f"[hrrr] restored {restored} samples from existing HRRR checkpoints", flush=True)

    periods = dates.to_period("M")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for month in periods.unique():
            month_indices = np.flatnonzero(periods == month)
            todo = pending_indices(status, month_indices, retry_failed=retry_failed)
            known_failed = int(np.count_nonzero(np.asarray(status[month_indices]) == 2))
            if not todo:
                saved = write_month_shard(
                    shard_dir,
                    month,
                    raw,
                    status,
                    dates,
                    month_indices,
                    latitude_patch,
                    longitude_patch,
                    min_required,
                )
                print(f"{month}: already attempted (saved {saved}, failed {known_failed})", flush=True)
                continue
            print(f"{month}: fetching {len(todo)} missing samples, skipping {known_failed} known failures", flush=True)
            tasks = ((index, dates[index].isoformat(), y0, x0, str(grib_dir)) for index in todo)
            successful = 0
            failed = 0
            changed = 0
            failure_counts = collections.Counter()
            suppressed_failures = 0
            for index, patch, error in executor.map(fetch_one, tasks, chunksize=1):
                if patch is None:
                    status[index] = 2
                    failed += 1
                    compact_error = compact_fetch_error(error)
                    failure_counts[compact_error] += 1
                    if failure_counts[compact_error] <= max_failure_logs:
                        print(f"Failed {dates[index]}: {compact_error}", flush=True)
                    else:
                        suppressed_failures += 1
                else:
                    raw[index, 0] = patch
                    status[index] = 1
                    successful += 1
                changed += 1
                if changed % max(1, flush_every) == 0:
                    raw.flush()
                    status.flush()
            raw.flush()
            status.flush()
            if suppressed_failures:
                print(f"{month}: suppressed {suppressed_failures} repeated failure logs", flush=True)
            saved = write_month_shard(
                shard_dir,
                month,
                raw,
                status,
                dates,
                month_indices,
                latitude_patch,
                longitude_patch,
                min_required,
            )
            print(f"{month}: downloaded {successful}, unavailable {failed}, saved {saved}", flush=True)

    successful_indices = np.flatnonzero(np.asarray(status) == 1)
    if len(successful_indices) < min_required:
        raise RuntimeError(f"Only {len(successful_indices)} HRRR samples were downloaded; required at least {min_required}.")

    compact_path = temporary_dir / "compact.npy"
    compact = np.lib.format.open_memmap(
        compact_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(successful_indices), 1, PATCH_SIZE, PATCH_SIZE),
    )
    for offset in range(0, len(successful_indices), 512):
        indices = successful_indices[offset:offset + 512]
        compact[offset:offset + len(indices)] = raw[indices]
    compact.flush()

    initialization_time = dates.asi8[successful_indices] // 1_000_000_000
    frames = torch.from_numpy(compact)
    save_hrrr_payload(
        output_path,
        frames,
        initialization_time,
        latitude_patch,
        longitude_patch,
        complete=True,
        min_required=min_required,
    )

    del frames, compact, raw, status
    if not keep_temp:
        shutil.rmtree(temporary_dir)
    print(f"Saved {len(successful_indices)} samples with shape ({len(successful_indices)}, 1, 100, 100) to {output_path}", flush=True)
    return int(len(successful_indices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Default: $WORK/hrrr_data/hrrr_apcp_100x100.pt")
    parser.add_argument("--start", default=START.isoformat())
    parser.add_argument("--end", default=None, help="Default: current UTC hour minus two hours.")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--flush-every", type=int, default=DEFAULT_FLUSH_EVERY, help="Flush raw/status memmaps after this many fetched samples.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry timestamps already marked unavailable in the warmstart status file.")
    parser.add_argument("--max-failure-logs", type=int, default=DEFAULT_MAX_FAILURE_LOGS, help="Maximum repeated failure lines to print per month and error type.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the existing file is large enough.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary memmaps and GRIB cache after success.")
    parser.add_argument("--status-file", type=Path, default=None, help="Optional JSON status path for Slurm wrappers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = parse_timestamp(args.start)
    end = default_end() if args.end is None else parse_timestamp(args.end)
    output_path = args.output.expanduser() if args.output is not None else default_output_path()
    min_required = required_samples(start, end, args.min_samples, args.min_coverage)
    current = count_hrrr_samples(output_path)
    print(f"[hrrr] current={current} required={min_required} path={output_path}", flush=True)

    if current >= min_required and not args.force:
        write_status(args.status_file, {"status": "exists", "samples": current, "required_samples": min_required, "path": str(output_path)})
        return 0
    if args.check_only:
        write_status(args.status_file, {"status": "missing", "samples": current, "required_samples": min_required, "path": str(output_path)})
        return 2

    samples = build_hrrr_dataset(
        start=start,
        end=end,
        output_path=output_path,
        n_workers=int(args.workers),
        min_required=min_required,
        keep_temp=bool(args.keep_temp),
        flush_every=int(args.flush_every),
        retry_failed=bool(args.retry_failed),
        max_failure_logs=int(args.max_failure_logs),
    )
    write_status(args.status_file, {"status": "built", "samples": samples, "required_samples": min_required, "path": str(output_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
