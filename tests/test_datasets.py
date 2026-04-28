"""Dataset module unittests."""

import json
import numpy as np
import os
from types import SimpleNamespace
from datetime import datetime
from sklearn.utils import Bunch
import pandas as pd
import pytest
import torch
from genkit._datasets import (
    DatasetPayload,
    _load_default_credit, _load_earthquakes, _load_hrrr, _load_lvis, _load_wildfires,
    _hrrr_cache_path,
    _resolve_real_data_home, _decode_byte_string, _standardize_split_arrays,
    _resolve_dataset, split_frame, split_tensor_data, _cache_remote_text_file,
    ALL_DATASETS,
)
from genkit.datasets import fetch_real_data, fetch_synthetic_data, get_dataset_metadata, list_datasets


def _mock_default_credit_bunch():
    frame = pd.DataFrame(
        {
            "ID": list(range(1, 11)),
            "LIMIT_BAL": [20_000, 120_000, 90_000, 50_000, 50_000, 80_000, 200_000, 30_000, 70_000, 150_000],
            "PAY_0": ["0", "1", "0", "-1", "0", "2", "0", "0", "-1", "1"],
            "SEX": ["female", "female", "male", "female", "male", "male", "female", "male", "female", "male"],
            "default payment next month": [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        }
    )
    return Bunch(
        frame=frame,
        data=frame.drop(columns=["default payment next month"]),
        target=frame["default payment next month"],
        target_names="default payment next month",
    )


def _write_lvis_fixture(root):
    image_mod = pytest.importorskip("PIL.Image")
    annotations_dir = root / "annotations"
    image_dir = root / "coco" / "train2017"
    annotations_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    images = []
    for image_id in range(1, 7):
        file_name = f"{image_id:012d}.jpg"
        color = (image_id * 30 % 255, image_id * 40 % 255, image_id * 50 % 255)
        image_mod.new("RGB", (10, 12), color=color).save(image_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "coco_url": f"http://images.cocodataset.org/train2017/{file_name}",
            }
        )

    payload = {
        "images": images,
        "categories": [
            {"id": 1, "name": "rare thing", "frequency": "r"},
            {"id": 2, "name": "frequent thing", "frequency": "f"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 1, "category_id": 1},
            {"id": 3, "image_id": 2, "category_id": 2},
            {"id": 4, "image_id": 3, "category_id": 1},
            {"id": 5, "image_id": 4, "category_id": 2},
            {"id": 6, "image_id": 5, "category_id": 1},
            {"id": 7, "image_id": 6, "category_id": 2},
        ],
    }
    (annotations_dir / "lvis_v1_train.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_load_default_credit_drops_identifier_and_coerces_numeric(monkeypatch):
    monkeypatch.setattr("genkit._datasets.fetch_openml", lambda **_: _mock_default_credit_bunch())

    frame = _load_default_credit()

    assert list(frame.columns) == ["LIMIT_BAL", "PAY_0", "SEX"]
    assert pd.api.types.is_numeric_dtype(frame["LIMIT_BAL"])
    assert pd.api.types.is_numeric_dtype(frame["PAY_0"])
    assert not pd.api.types.is_numeric_dtype(frame["SEX"])


def test_fetch_real_data_supports_default_credit(monkeypatch):
    monkeypatch.setattr("genkit._datasets.fetch_openml", lambda **_: _mock_default_credit_bunch())

    x_train, x_val, x_test = fetch_real_data(
        "default_credit",
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )

    assert x_train.shape == (6, 4)
    assert x_val.shape == (2, 4)
    assert x_test.shape == (2, 4)
    assert x_train.dtype == torch.float64
    assert x_val.dtype == torch.float64
    assert x_test.dtype == torch.float64


def test_default_credit_is_registered():
    assert "default_credit" in list_datasets()
    assert get_dataset_metadata("default_credit") == {
        "name": "default_credit",
        "tail_index_alpha": None,
        "description": "Default of Credit Card Clients dataset from OpenML.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": None,
        "n_samples": None,
    }


def test_resolve_real_data_home_prefers_work(tmp_path, monkeypatch):
    work_root = tmp_path / "work"
    home_root = tmp_path / "home"
    monkeypatch.setenv("WORK", str(work_root))
    monkeypatch.setenv("HOME", str(home_root))

    resolved = _resolve_real_data_home()

    assert resolved == work_root / "flowbench_data"


def test_resolve_real_data_home_uses_home_when_work_is_missing(tmp_path, monkeypatch):
    home_root = tmp_path / "home"
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.setenv("HOME", str(home_root))

    resolved = _resolve_real_data_home()

    assert resolved == home_root / ".cache" / "flowbench_data"


def test_load_earthquakes_uses_cached_file(tmp_path, monkeypatch):
    data_home = tmp_path / "powerlaws"
    data_home.mkdir()
    (data_home / "quakes.txt").write_text("1.0\n2.5\n", encoding="utf-8")
    monkeypatch.setattr("genkit._datasets.urlretrieve", lambda *args, **kwargs: pytest.fail("cache should bypass download"))

    frame = _load_earthquakes(data_home=data_home)

    assert list(frame["magnitude"]) == [1.0, 2.5]


def test_load_wildfires_uses_cached_file(tmp_path, monkeypatch):
    data_home = tmp_path / "powerlaws"
    data_home.mkdir()
    (data_home / "fires.txt").write_text("10\n20\n", encoding="utf-8")
    monkeypatch.setattr("genkit._datasets.urlretrieve", lambda *args, **kwargs: pytest.fail("cache should bypass download"))

    frame = _load_wildfires(data_home=data_home)

    assert list(frame["acres_burned"]) == [10.0, 20.0]


def test_load_hrrr_saves_and_reuses_cache(tmp_path, monkeypatch):
    values = np.arange(10000, dtype=np.float32).reshape(100, 100)

    class _FakeVar:
        def __init__(self, array):
            self._array = array

        def squeeze(self, drop=True):
            return self

        @property
        def values(self):
            return self._array

    class _FakeDataset:
        def __init__(self, array):
            self.data_vars = {"apcp": _FakeVar(array)}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    _FAKE_IDX = "1:0:d=2014100100:APCP:surface:0-1 hour acc fcst:\n2:9999:d=2014100100:TMP:2 m above ground:\n"

    class _FakeResponse:
        def __init__(self, text="", content=b""):
            self.text = text
            self.ok = True
            self.status_code = 200
            self._content = content

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=None):
            yield self._content

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            if url.endswith(".idx"):
                return _FakeResponse(text=_FAKE_IDX)
            return _FakeResponse(content=b"")

    subprocess_calls = []

    def _fake_run(*args, **kwargs):
        cmd = args[0]
        subprocess_calls.append(cmd)
        import subprocess as _sp
        return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    _fake_session = _FakeSession()
    monkeypatch.setattr("genkit._datasets.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("genkit._datasets.requests.Session", _FakeSession)
    monkeypatch.setattr("genkit._datasets._hrrr_session", lambda: _fake_session)
    monkeypatch.setattr("genkit._datasets.subprocess.run", _fake_run)
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: _FakeDataset(values))

    loaded = _load_hrrr(
        data_home=tmp_path,
        start="2014-10-01 00:00:00",
        end="2014-10-01 02:00:00",
    )
    cache_paths = list(tmp_path.glob("hrrr_apcp_100x100_*.pt"))

    assert loaded.shape == (3, 1, 100, 100)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded[0, 0], torch.from_numpy(values))
    assert len(cache_paths) == 1
    assert len(subprocess_calls) == 3  # per sample: wgrib2 only

    monkeypatch.setattr("genkit._datasets.subprocess.run", lambda *args, **kwargs: pytest.fail("cache should bypass download"))
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: pytest.fail("cache should bypass decoding"))
    cached = _load_hrrr(data_home=tmp_path, start="2014-10-01 00:00:00", end="2014-10-01 02:00:00")

    assert torch.equal(cached, loaded)


def test_load_hrrr_resumes_from_partial_cache(tmp_path, monkeypatch):
    values = np.arange(10000, dtype=np.float32).reshape(100, 100)

    class _FakeVar:
        def __init__(self, array):
            self._array = array

        def squeeze(self, drop=True):
            return self

        @property
        def values(self):
            return self._array

    class _FakeDataset:
        def __init__(self, array):
            self.data_vars = {"apcp": _FakeVar(array)}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    _FAKE_IDX = "1:0:d=2014100100:APCP:surface:0-1 hour acc fcst:\n2:9999:d=2014100100:TMP:2 m above ground:\n"

    class _FakeResponse:
        def __init__(self, text="", content=b""):
            self.text = text
            self.ok = True
            self.status_code = 200
            self._content = content

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=None):
            yield self._content

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            if url.endswith(".idx"):
                return _FakeResponse(text=_FAKE_IDX)
            return _FakeResponse(content=b"")

    subprocess_calls = []

    def _fake_run(*args, **kwargs):
        cmd = args[0]
        subprocess_calls.append(cmd)
        import subprocess as _sp
        return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    cache_root = tmp_path
    cache_path = _hrrr_cache_path(
        cache_root,
        start=datetime(2014, 10, 1, 0),
        end=datetime(2014, 10, 1, 2),
        bbox=(-96.0, -91.5, 28.5, 32.0),
        step_hours=1,
        forecast_hours=(1,),
    )
    torch.save(
        {
            "timestamps": ["2014-10-01T00:00:00Z"],
            "frames": torch.from_numpy(values).reshape(1, 1, 100, 100),
        },
        cache_path,
    )

    _fake_session = _FakeSession()
    monkeypatch.setattr("genkit._datasets.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("genkit._datasets.requests.Session", _FakeSession)
    monkeypatch.setattr("genkit._datasets._hrrr_session", lambda: _fake_session)
    monkeypatch.setattr("genkit._datasets.subprocess.run", _fake_run)
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: _FakeDataset(values))

    loaded = _load_hrrr(
        data_home=cache_root,
        start="2014-10-01 00:00:00",
        end="2014-10-01 02:00:00",
    )

    assert loaded.shape == (3, 1, 100, 100)
    assert len(subprocess_calls) == 2
    assert cache_path.exists()


def test_load_hrrr_returns_immediately_when_cache_has_requested_n_samples(tmp_path, monkeypatch):
    values = np.arange(10000, dtype=np.float32).reshape(100, 100)

    cache_path = _hrrr_cache_path(
        tmp_path,
        start=datetime(2014, 10, 1, 0),
        end=datetime(2014, 10, 1, 2),
        bbox=(-96.0, -91.5, 28.5, 32.0),
        step_hours=1,
        forecast_hours=(1,),
    )
    torch.save(
        {
            "timestamps": ["2014-10-01T00:00:00Z", "2014-10-01T01:00:00Z"],
            "frames": torch.from_numpy(np.stack([values, values + 1], axis=0)[:, np.newaxis]),
        },
        cache_path,
    )

    monkeypatch.setattr("genkit._datasets.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("genkit._datasets.subprocess.run", lambda *args, **kwargs: pytest.fail("should not fetch more frames"))
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: pytest.fail("should not fetch more frames"))

    loaded = _load_hrrr(
        data_home=tmp_path,
        start="2014-10-01 00:00:00",
        end="2014-10-01 02:00:00",
        n_samples=2,
    )

    assert loaded.shape == (2, 1, 100, 100)
    assert torch.equal(loaded[0, 0], torch.from_numpy(values))
    assert torch.equal(loaded[1, 0], torch.from_numpy(values + 1))
    assert cache_path.exists()


def test_load_hrrr_fetches_only_missing_target_samples(tmp_path, monkeypatch):
    values = np.arange(10000, dtype=np.float32).reshape(100, 100)

    class _FakeVar:
        def __init__(self, array):
            self._array = array

        def squeeze(self, drop=True):
            return self

        @property
        def values(self):
            return self._array

    class _FakeDataset:
        def __init__(self, array):
            self.data_vars = {"apcp": _FakeVar(array)}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeResponse:
        text = "1:0:d=2014100100:APCP:surface:0-1 hour acc fcst:\n2:9999:d=2014100100:TMP:2 m above ground:\n"
        ok = True
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=None):
            yield b""

    class _FakeSession:
        def get(self, url, **kwargs):
            return _FakeResponse()

    cache_path = _hrrr_cache_path(
        tmp_path,
        start=datetime(2014, 10, 1, 0),
        end=datetime(2014, 10, 1, 2),
        bbox=(-96.0, -91.5, 28.5, 32.0),
        step_hours=1,
        forecast_hours=(1,),
    )
    torch.save(
        {
            "timestamps": ["2014-10-01T00:00:00Z"],
            "frames": torch.from_numpy(values).reshape(1, 1, 100, 100),
        },
        cache_path,
    )

    subprocess_calls = []

    def _fake_run(*args, **kwargs):
        subprocess_calls.append(args[0])
        import subprocess as _sp
        return _sp.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr("genkit._datasets.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("genkit._datasets._hrrr_session", lambda: _FakeSession())
    monkeypatch.setattr("genkit._datasets.subprocess.run", _fake_run)
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: _FakeDataset(values + 1))

    loaded = _load_hrrr(
        data_home=tmp_path,
        start="2014-10-01 00:00:00",
        end="2014-10-01 02:00:00",
        n_samples=2,
        n_workers=8,
    )

    assert loaded.shape == (2, 1, 100, 100)
    assert len(subprocess_calls) == 1


def test_fetch_real_data_supports_hrrr_tensor_dataset(tmp_path):
    def _fake_loader(**kwargs):
        assert kwargs["data_home"] == tmp_path
        return torch.arange(5 * 1 * 3 * 4, dtype=torch.float32).reshape(5, 1, 3, 4)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "genkit.datasets._dataset._resolve_dataset",
        lambda target_data, all_datasets, dataset_type=None: SimpleNamespace(
            loader=_fake_loader,
            split_mode="random",
            dataset_type="real",
            standardize_default=False,
        ),
    )

    x_train, x_val, x_test = fetch_real_data(
        "hrrr",
        data_home=tmp_path,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )
    monkeypatch.undo()

    assert x_train.shape == (3, 1, 3, 4)
    assert x_val.shape == (1, 1, 3, 4)
    assert x_test.shape == (1, 1, 3, 4)
    assert x_train.dtype == torch.float64
    assert x_val.dtype == torch.float64
    assert x_test.dtype == torch.float64


def test_hrrr_dataset_is_registered():
    assert "hrrr" in list_datasets()
    assert get_dataset_metadata("hrrr") == {
        "name": "hrrr",
        "tail_index_alpha": None,
        "description": "HRRR accumulated precipitation fields on a 100x100 crop.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": (1, 100, 100),
        "n_samples": None,
    }


def test_load_lvis_reads_local_fixture_and_filters_frequency(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    loaded = _load_lvis(
        data_home=data_home,
        split="train",
        image_size=8,
        max_samples=10,
        category_frequency="r",
    )

    assert isinstance(loaded, DatasetPayload)
    tensor = loaded.data
    assert tensor.shape == (3, 3, 8, 8)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert loaded.metadata["source_split"] == "train"
    assert loaded.metadata["category_frequency_filter"] == "r"
    assert loaded.metadata["n_selected_images"] == 3
    assert [record["image_id"] for record in loaded.metadata["records"]] == [1, 3, 5]
    assert all(record["category_frequencies"] == ["r"] for record in loaded.metadata["records"])


def test_fetch_real_data_supports_lvis_tensor_dataset(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    x_train, x_val, x_test = fetch_real_data(
        "lvis",
        data_home=data_home,
        split="train",
        image_size=8,
        max_samples=5,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
    )

    assert x_train.shape == (3, 3, 8, 8)
    assert x_val.shape == (1, 3, 8, 8)
    assert x_test.shape == (1, 3, 8, 8)
    assert x_train.dtype == torch.float32


def test_fetch_real_data_passes_lvis_n_samples_as_max_samples(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    x_train, x_val, x_test = fetch_real_data(
        "lvis",
        data_home=data_home,
        split="train",
        image_size=8,
        n_samples=5,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
    )

    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == 5


def test_fetch_real_data_return_metadata_includes_lvis_records_and_histograms(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    x_train, x_val, x_test, metadata = fetch_real_data(
        "lvis",
        data_home=data_home,
        split="train",
        image_size=8,
        max_samples=5,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
        return_metadata=True,
    )

    assert x_train.shape == (3, 3, 8, 8)
    assert x_val.shape == (1, 3, 8, 8)
    assert x_test.shape == (1, 3, 8, 8)
    assert metadata["dataset"] == get_dataset_metadata("lvis")
    assert metadata["request"]["name"] == "lvis"
    assert metadata["request"]["params"]["max_samples"] == 5
    assert metadata["loader"]["image_size"] == 8
    assert metadata["loader"]["n_selected_images"] == 5

    split_records = [
        record
        for split_name in ("train", "val", "test")
        for record in metadata["splits"][split_name]["records"]
    ]
    assert len(split_records) == 5
    assert len({record["image_id"] for record in split_records}) == 5
    assert sum(metadata["splits"][name]["n_samples"] for name in ("train", "val", "test")) == 5
    assert all("category_histogram" in metadata["splits"][name] for name in ("train", "val", "test"))
    assert all("frequency_histogram" in metadata["splits"][name] for name in ("train", "val", "test"))


def test_lvis_uses_dsdir_and_fails_clearly_when_absent(monkeypatch):
    monkeypatch.delenv("DSDIR", raising=False)

    with pytest.raises(RuntimeError, match="DSDIR"):
        _load_lvis(max_samples=1)


def test_lvis_dataset_is_registered():
    assert "lvis" in list_datasets()
    assert get_dataset_metadata("lvis") == {
        "name": "lvis",
        "tail_index_alpha": None,
        "description": "LVIS long-tailed object categories from local Jean Zay COCO/LVIS files; default image_size=64.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": (3, 64, 64),
        "n_samples": None,
    }


def test_lvis_smoke_from_dsdir_if_available():
    if not os.getenv("DSDIR"):
        pytest.skip("Jean Zay $DSDIR is not set.")
    try:
        loaded = _load_lvis(split="train", image_size=8, max_samples=3, seed=0)
    except RuntimeError as exc:
        pytest.skip(f"LVIS assets are not available under $DSDIR: {exc}")

    assert isinstance(loaded, DatasetPayload)
    tensor = loaded.data
    assert tensor.shape == (3, 3, 8, 8)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert len(loaded.metadata["records"]) == 3


def test_kddcup_dataset_is_registered():
    assert "kddcup" in list_datasets()
    assert get_dataset_metadata("kddcup") == {
        "name": "kddcup",
        "tail_index_alpha": None,
        "description": "KDD Cup 99 intrusion dataset with numeric feature columns only.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": None,
        "n_samples": None,
    }


def test_fetch_synthetic_data_supports_gaussian():
    x_train, x_val, x_test = fetch_synthetic_data(
        "gaussian",
        n_samples=20,
        dim=3,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float32,
    )

    assert x_train.shape == (12, 3)
    assert x_val.shape == (4, 3)
    assert x_test.shape == (4, 3)
    assert x_train.dtype == torch.float32


def test_fetch_synthetic_data_return_metadata_includes_dataset_and_splits():
    x_train, x_val, x_test, metadata = fetch_synthetic_data(
        "gaussian",
        n_samples=20,
        dim=3,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float32,
        return_metadata=True,
    )

    assert x_train.shape == (12, 3)
    assert x_val.shape == (4, 3)
    assert x_test.shape == (4, 3)
    assert metadata["dataset"] == get_dataset_metadata("gaussian")
    assert metadata["request"]["kind"] == "synthetic"
    assert metadata["request"]["params"]["n_samples"] == 20
    assert metadata["request"]["params"]["dim"] == 3
    assert sum(metadata["splits"][name]["n_samples"] for name in ("train", "val", "test")) == 20


def test_fetch_synthetic_data_supports_unbalanced_highdim_gaussian_mixture():
    x_train, x_val, x_test = fetch_synthetic_data(
        "unbalanced_highdim_gaussian_mixture",
        n_samples=40,
        dim=10,
        n_modes=8,
        rank=3,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )

    assert x_train.shape == (24, 10)
    assert x_val.shape == (8, 10)
    assert x_test.shape == (8, 10)
    assert x_train.dtype == torch.float64


def test_fetch_synthetic_data_supports_unbalanced_highdim_alpha_stable_mixture():
    x_train, x_val, x_test = fetch_synthetic_data(
        "unbalanced_highdim_alpha_stable_mixture",
        n_samples=40,
        dim=10,
        alpha=1.6,
        n_modes=8,
        rank=3,
        base_std=0.35,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )

    assert x_train.shape == (24, 10)
    assert x_val.shape == (8, 10)
    assert x_test.shape == (8, 10)
    assert x_train.dtype == torch.float64


def test_bimodal_gaussian_datasets_are_not_registered():
    assert "balanced_bimodal_gaussian" not in list_datasets()
    assert "unbalanced_bimodal_gaussian" not in list_datasets()
    with pytest.raises(KeyError, match="balanced_bimodal_gaussian"):
        get_dataset_metadata("balanced_bimodal_gaussian")
    with pytest.raises(KeyError, match="unbalanced_bimodal_gaussian"):
        fetch_synthetic_data("unbalanced_bimodal_gaussian", n_samples=8)


def test_fetch_synthetic_data_rejects_real_dataset_name():
    with pytest.raises(ValueError, match="expected 'synthetic'"):
        fetch_synthetic_data("default_credit", n_samples=8)


def test_fetch_synthetic_data_unknown_name_raises_key_error():
    with pytest.raises(KeyError, match="no_such_dataset"):
        fetch_synthetic_data("no_such_dataset")


def test_decode_byte_string_decodes_bytes_to_str():
    assert _decode_byte_string(b"hello") == "hello"


def test_decode_byte_string_passes_through_non_bytes():
    assert _decode_byte_string("hello") == "hello"
    assert _decode_byte_string(42) == 42


def test_resolve_real_data_home_flowbench_data_home_takes_precedence(tmp_path, monkeypatch):
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("FLOWBENCH_DATA_HOME", str(custom))
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    resolved = _resolve_real_data_home()
    assert resolved == custom
    assert custom.exists()


def test_split_frame_chronological_preserves_temporal_order():
    frame = pd.DataFrame({"x": list(range(20))})
    train, val, test = split_frame(
        frame, val_size=0.2, test_size=0.2, random_state=0, split_mode="chronological"
    )
    assert int(train[-1, 0]) < int(val[0, 0])
    assert int(val[-1, 0]) < int(test[0, 0])


def test_split_frame_chronological_too_small_raises():
    # 3 rows with val/test=0.2 each: floor(0.2*3)=0 → empty val/test split
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="empty split"):
        split_frame(frame, val_size=0.2, test_size=0.2, random_state=0, split_mode="chronological")


def test_split_frame_invalid_split_mode_raises():
    frame = pd.DataFrame({"x": list(range(10))})
    with pytest.raises(ValueError, match="split_mode"):
        split_frame(frame, val_size=0.2, test_size=0.2, random_state=0, split_mode="bad_mode")


def test_standardize_split_arrays_zero_variance_column_does_not_produce_nan():
    train = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    val = np.array([[1.0, 3.0]])
    test = np.array([[1.0, -1.0]])
    tr, v, te = _standardize_split_arrays(train, val, test)
    assert np.isfinite(tr).all()
    assert np.isfinite(v).all()
    assert np.isfinite(te).all()
    # constant column → mean=1, std clamped to 1 → (1-1)/1 = 0
    assert (tr[:, 0] == 0.0).all()


def test_resolve_dataset_type_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="expected 'real'"):
        _resolve_dataset("gaussian", ALL_DATASETS, dataset_type="real")


def test_split_tensor_data_chronological_preserves_leading_order():
    data = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    train, val, test = split_tensor_data(
        data, val_size=0.2, test_size=0.2, random_state=0, split_mode="chronological"
    )
    assert float(train[-1, 0]) < float(val[0, 0])
    assert float(val[-1, 0]) < float(test[0, 0])


def test_fetch_real_data_n_samples_exceeding_available_warns(monkeypatch):
    small_df = pd.DataFrame({"x": list(range(6)), "y": list(range(6, 12))})
    fake_entry = SimpleNamespace(
        loader=lambda **kw: small_df,
        split_mode="random",
        dataset_type="real",
        standardize_default=False,
    )
    monkeypatch.setattr("genkit.datasets._dataset._resolve_dataset", lambda *a, **kw: fake_entry)
    with pytest.warns(UserWarning, match="exceeds available"):
        fetch_real_data("anything", n_samples=100, val_size=0.2, test_size=0.2)


def test_cache_remote_text_file_cache_hit_skips_download(tmp_path, monkeypatch):
    cached = tmp_path / "data.txt"
    cached.write_text("1.0\n2.0\n")
    monkeypatch.setattr(
        "genkit._datasets.urlretrieve",
        lambda *a, **kw: pytest.fail("should not download when cached"),
    )
    result = _cache_remote_text_file("http://example.com/data.txt", data_home=tmp_path, filename="data.txt")
    assert result == cached
