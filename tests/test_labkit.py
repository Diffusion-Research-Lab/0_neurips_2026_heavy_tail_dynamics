"""Tests for labkit config and reporting helpers."""

from pathlib import Path
import textwrap
import random
import pytest
import torch

from labkit.config import load_config, parse_dtype, parse_idtype
from labkit.report import format_mean_std_latex, summarize_metric_values, to_latex_sci
from labkit.utils import get_device, set_seed


def test_parse_dtype_and_idtype_accept_common_aliases():
    assert parse_dtype("float") is torch.float32
    assert parse_dtype("double") is torch.float64
    assert parse_dtype("half") is torch.float16
    assert parse_idtype("int") is torch.int32
    assert parse_idtype("long") is torch.int64


def test_parse_dtype_and_idtype_reject_unknown_values():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        parse_dtype("complex64")
    with pytest.raises(ValueError, match="Unsupported integer dtype"):
        parse_idtype("uint8")


def test_load_config_materializes_special_fields_and_runtime_types(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            dtype: float32
            fdtype: float64
            idtype: long
            device: cpu
            seed: 3
            alpha_grid:
              type: linspace
              start: 1.2
              end: 1.8
              steps: 4
            tensor_values:
              type: tensor
              values: [1, 2, 3]
            authors_root:
              mode: explicit
              path: {authors_root}
              env_var: FLOWBENCH_AUTHORS_ROOT
            """.format(authors_root=tmp_path.resolve())
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.dtype is torch.float32
    assert cfg.fdtype is torch.float64
    assert cfg.idtype is torch.int64
    assert cfg.device == torch.device("cpu")
    assert torch.allclose(cfg.alpha_grid, torch.linspace(1.2, 1.8, 4, dtype=torch.float32))
    assert torch.equal(cfg.tensor_values, torch.tensor([1, 2, 3], dtype=torch.float32))
    assert Path(cfg.AUTHORS_ROOT) == tmp_path.resolve()
    assert cfg.authors_root_env_var == "FLOWBENCH_AUTHORS_ROOT"


def test_load_config_set_up_applies_runtime_side_effects(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            seed: 11
            device: cpu
            dtype: float32
            authors_root:
              mode: explicit
              path: {authors_root}
              env_var: FLOWBENCH_AUTHORS_ROOT
            """.format(authors_root=tmp_path.resolve())
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path).set_up()

    assert cfg.device == torch.device("cpu")
    assert torch.get_default_dtype() is torch.float32
    assert Path(cfg.AUTHORS_ROOT) == tmp_path.resolve()


def test_load_config_rejects_unknown_authors_root_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            authors_root:
              mode: unknown
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authors_root mode"):
        load_config(config_path)


def test_report_helpers_format_compact_scientific_strings():
    stats = summarize_metric_values([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx((2.0 / 3.0) ** 0.5)

    assert to_latex_sci(0.0123) == r"1.23\,10^{-2}"
    assert to_latex_sci(2.5) == "2.50"

    plain = format_mean_std_latex(2.5, 0.0)
    captioned = format_mean_std_latex(1.23e-3, 4.56e-4, caption="S", bold=True)
    assert plain == r"$2.50$"
    assert r"\underset" in captioned
    assert r"\mathbf" in captioned
    assert r"\pm" in captioned


def test_set_seed_reproducibly_resets_python_and_torch_rngs():
    set_seed(7)
    py_1 = random.random()
    torch_1 = torch.randn(3)

    set_seed(7)
    py_2 = random.random()
    torch_2 = torch.randn(3)

    assert py_1 == py_2
    assert torch.allclose(torch_1, torch_2)


def test_get_device_supports_auto_and_explicit_cpu():
    assert get_device("cpu") == torch.device("cpu")
    assert isinstance(get_device("auto"), torch.device)
