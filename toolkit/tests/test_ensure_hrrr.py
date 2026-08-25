"""Tests for resumable HRRR raw-tensor construction helpers."""

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


def load_ensure_hrrr_module():
    path = Path(__file__).resolve().parents[2] / "setup" / "ensure_hrrr.py"
    spec = importlib.util.spec_from_file_location("ensure_hrrr", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_output_path_uses_store(tmp_path, monkeypatch):
    ensure_hrrr = load_ensure_hrrr_module()
    monkeypatch.setenv("STORE", str(tmp_path))

    assert ensure_hrrr.default_output_path() == tmp_path / "hrrr_data" / "hrrr_apcp_100x100.pt"


def test_hrrr_month_shard_restores_into_fresh_memmaps(tmp_path):
    ensure_hrrr = load_ensure_hrrr_module()
    dates = pd.date_range("2020-01-01", periods=3, freq="h")
    shard_dir = tmp_path / ensure_hrrr.SHARD_DIR_NAME

    raw, status = ensure_hrrr.open_or_create_memmaps(
        tmp_path / "raw.npy",
        tmp_path / "status.npy",
        (3, 1, ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE),
    )
    raw[0] = 1.0
    raw[2] = 3.0
    status[[0, 2]] = 1
    raw.flush()
    status.flush()

    latitude = np.zeros((ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE), dtype=np.float32)
    longitude = np.ones((ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE), dtype=np.float32)
    saved = ensure_hrrr.write_month_shard(
        shard_dir,
        dates.to_period("M")[0],
        raw,
        status,
        dates,
        np.arange(3),
        latitude,
        longitude,
        min_required=2,
    )
    assert saved == 2

    restored_raw, restored_status = ensure_hrrr.open_or_create_memmaps(
        tmp_path / "restored_raw.npy",
        tmp_path / "restored_status.npy",
        (3, 1, ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE),
    )
    restored = ensure_hrrr.restore_checkpoints_into_memmaps(
        tmp_path / "missing_final.pt",
        shard_dir,
        dates,
        restored_raw,
        restored_status,
    )

    assert restored == 2
    assert restored_status.tolist() == [1, 0, 1]
    assert np.all(restored_raw[0] == 1.0)
    assert np.all(restored_raw[2] == 3.0)


def test_hrrr_memmaps_extend_for_newer_end_date(tmp_path):
    ensure_hrrr = load_ensure_hrrr_module()
    raw_path = tmp_path / "raw.npy"
    status_path = tmp_path / "status.npy"

    raw, status = ensure_hrrr.open_or_create_memmaps(
        raw_path,
        status_path,
        (2, 1, ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE),
    )
    raw[0] = 5.0
    status[0] = 1
    raw.flush()
    status.flush()
    del raw, status

    raw, status = ensure_hrrr.open_or_create_memmaps(
        raw_path,
        status_path,
        (4, 1, ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE),
    )

    assert raw.shape == (4, 1, ensure_hrrr.PATCH_SIZE, ensure_hrrr.PATCH_SIZE)
    assert status.tolist() == [1, 0, 0, 0]
    assert np.all(raw[0] == 5.0)


def test_hrrr_pending_indices_skip_known_failures_by_default():
    ensure_hrrr = load_ensure_hrrr_module()
    status = np.array([0, 1, 2, 0, 2], dtype=np.uint8)
    indices = np.arange(len(status))

    assert ensure_hrrr.pending_indices(status, indices, retry_failed=False) == [0, 3]
    assert ensure_hrrr.pending_indices(status, indices, retry_failed=True) == [0, 2, 3, 4]


def test_hrrr_compact_fetch_error_shortens_common_herbie_errors():
    ensure_hrrr = load_ensure_hrrr_module()

    assert ensure_hrrr.compact_fetch_error("ValueError('\\nNo index file was found for None\\n...')") == "missing Herbie index"
    assert ensure_hrrr.compact_fetch_error("ProxyError(MaxRetryError('bad gateway'))") == "proxy/network error"
