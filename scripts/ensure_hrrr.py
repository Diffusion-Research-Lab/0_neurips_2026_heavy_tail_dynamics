#!/usr/bin/env python
"""Ensure the raw HRRR APCP tensor has enough hourly samples."""

import argparse
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
from herbie import Herbie


START = pd.Timestamp("2014-07-30 00:00:00")
PATCH_SIZE = 100
CENTER_LAT = 39.0
CENTER_LON = -98.0
SEARCH = r":APCP:surface:0-1 hour acc fcst:"
DEFAULT_MIN_COVERAGE = 0.90


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


def build_hrrr_dataset(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
    n_workers: int,
    min_required: int,
    keep_temp: bool,
) -> int:
    import torch

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / ".hrrr_apcp_100x100_tmp"
    grib_dir = temporary_dir / "grib"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    grib_dir.mkdir(parents=True, exist_ok=True)

    config_path = temporary_dir / "config.json"
    latitude_path = temporary_dir / "latitude.npy"
    longitude_path = temporary_dir / "longitude.npy"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        start = parse_timestamp(config["start"])
        end = parse_timestamp(config["end"])
        y0 = int(config["y0"])
        x0 = int(config["x0"])
        latitude_patch = np.load(latitude_path)
        longitude_patch = np.load(longitude_path)
    else:
        y0, x0, latitude_patch, longitude_patch = find_patch(end, grib_dir)
        np.save(latitude_path, latitude_patch)
        np.save(longitude_path, longitude_patch)
        config = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "y0": y0,
            "x0": x0,
            "center_lat": CENTER_LAT,
            "center_lon": CENTER_LON,
            "patch_size": PATCH_SIZE,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    dates = pd.date_range(start, end, freq="h")
    n_possible = len(dates)
    raw_path = temporary_dir / "raw.npy"
    status_path = temporary_dir / "status.npy"
    expected_shape = (n_possible, 1, PATCH_SIZE, PATCH_SIZE)
    if raw_path.exists() and status_path.exists():
        raw = np.lib.format.open_memmap(raw_path, mode="r+")
        status = np.lib.format.open_memmap(status_path, mode="r+")
        if raw.shape != expected_shape or status.shape != (n_possible,):
            raise RuntimeError("Temporary files are incompatible with the saved HRRR configuration")
    else:
        raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=expected_shape)
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.uint8, shape=(n_possible,))
        status[:] = 0
        status.flush()

    periods = dates.to_period("M")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for month in periods.unique():
            month_indices = np.flatnonzero(periods == month)
            todo = [int(index) for index in month_indices if status[index] != 1]
            if not todo:
                print(f"{month}: already complete", flush=True)
                continue
            tasks = ((index, dates[index].isoformat(), y0, x0, str(grib_dir)) for index in todo)
            successful = 0
            failed = 0
            for index, patch, error in executor.map(fetch_one, tasks, chunksize=1):
                if patch is None:
                    status[index] = 2
                    failed += 1
                    print(f"Failed {dates[index]}: {error}", flush=True)
                else:
                    raw[index, 0] = patch
                    status[index] = 1
                    successful += 1
            raw.flush()
            status.flush()
            print(f"{month}: downloaded {successful}, unavailable {failed}", flush=True)

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
    dataset = {
        "frames": frames,
        "apcp_mm": frames,
        "init_time_unix_s": torch.from_numpy(initialization_time.copy()),
        "valid_time_unix_s": torch.from_numpy((initialization_time + 3600).copy()),
        "latitude": torch.from_numpy(latitude_patch.astype(np.float32, copy=False)),
        "longitude": torch.from_numpy(longitude_patch.astype(np.float32, copy=False)),
        "forecast_hour": 1,
        "units": "mm",
        "variable": "APCP",
        "center_latitude": CENTER_LAT,
        "center_longitude": CENTER_LON,
    }
    staged_path = temporary_dir / output_path.name
    torch.save(dataset, staged_path)
    os.replace(staged_path, output_path)

    del dataset, frames, compact, raw, status
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
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
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
    )
    write_status(args.status_file, {"status": "built", "samples": samples, "required_samples": min_required, "path": str(output_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
