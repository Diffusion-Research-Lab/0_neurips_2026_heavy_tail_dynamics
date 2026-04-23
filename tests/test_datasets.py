"""Dataset module unittests."""

import numpy as np
from types import SimpleNamespace
from sklearn.utils import Bunch
import pandas as pd
import pytest
import torch
from genkit._datasets import _load_default_credit, _load_earthquakes, _load_hrrr, _load_wildfires, _resolve_real_data_home
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

    assert resolved == work_root / "flowbench" / "data"


def test_resolve_real_data_home_uses_home_when_work_is_missing(tmp_path, monkeypatch):
    home_root = tmp_path / "home"
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.setenv("HOME", str(home_root))

    resolved = _resolve_real_data_home()

    assert resolved == home_root / ".cache" / "flowbench" / "data"


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

    _FAKE_IDX = "1:0:d=2014100100:APCP:surface:0-6 hour acc fcst:\n2:9999:d=2014100100:TMP:2 m above ground:\n"

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

    monkeypatch.setattr("genkit._datasets.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("genkit._datasets.requests.Session", _FakeSession)
    monkeypatch.setattr("genkit._datasets.subprocess.run", _fake_run)
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: _FakeDataset(values))

    loaded = _load_hrrr(
        data_home=tmp_path,
        start="2014-10-01 00:00:00",
        end="2014-10-01 06:00:00",
    )
    cache_path = tmp_path / "hrrr.pt"

    assert loaded.shape == (2, 1, 100, 100)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded[0, 0], torch.from_numpy(values))
    assert cache_path.exists()
    assert len(subprocess_calls) == 2  # per sample: wgrib2 only

    monkeypatch.setattr("genkit._datasets.subprocess.run", lambda *args, **kwargs: pytest.fail("cache should bypass download"))
    monkeypatch.setattr("genkit._datasets.xr.open_dataset", lambda *args, **kwargs: pytest.fail("cache should bypass decoding"))
    cached = _load_hrrr(data_home=tmp_path, start="2014-10-01 00:00:00", end="2014-10-01 06:00:00")

    assert torch.equal(cached, loaded)


def test_fetch_real_data_supports_hrrr_tensor_dataset(tmp_path):
    def _fake_loader(**kwargs):
        assert kwargs["data_home"] == tmp_path
        return torch.arange(5 * 1 * 3 * 4, dtype=torch.float32).reshape(5, 1, 3, 4)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "genkit.datasets._resolve_dataset",
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


def test_unbalanced_bimodal_gaussian_metadata_exposes_legacy_alias():
    assert get_dataset_metadata("unbalanced_bimodal_gaussian") == {
        "name": "unbalanced_bimodal_gaussian",
        "tail_index_alpha": None,
        "description": "Legacy alias for an imbalanced high-dimensional Gaussian mixture synthetic dataset.",
        "split_mode": "random",
        "dataset_type": "synthetic",
        "dim": None,
        "n_samples": None,
    }


def test_fetch_synthetic_data_rejects_real_dataset_name():
    with pytest.raises(ValueError, match="expected 'synthetic'"):
        fetch_synthetic_data("default_credit", n_samples=8)
