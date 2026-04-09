"""Dataset module unittests."""

from sklearn.utils import Bunch
import pandas as pd
import pytest
import torch

from genkit._datasets import _load_default_credit
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


def test_fetch_synthetic_data_rejects_real_dataset_name():
    with pytest.raises(ValueError, match="expected 'synthetic'"):
        fetch_synthetic_data("default_credit", n_samples=8)
