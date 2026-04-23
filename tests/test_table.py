"""Tests for LaTeX table helpers."""

from pathlib import Path
import pytest
import torch
from genkit.table import dict_to_double_entry_latex_table, save_double_entry_table


def _toy_results():
    return {
        ("b", "m2"): torch.tensor([2.0, 3.0]),
        ("a", "m1"): torch.tensor([1.0, 1.5]),
        ("a", "m2"): torch.tensor([0.5, 1.0]),
        ("b", "m1"): torch.tensor([1.2, 1.8]),
    }


def test_default_metric_direction_string_works():
    tex = dict_to_double_entry_latex_table(_toy_results())
    assert tex.count(r"$\downarrow$") == 2


def test_missing_row_col_order_infers_deterministic_lists():
    tex = dict_to_double_entry_latex_table(_toy_results())
    assert " & a & b \\\\" in tex
    assert "m1 " in tex
    assert "m2 " in tex


def test_callable_fmt_supported():
    tex = dict_to_double_entry_latex_table(
        _toy_results(),
        fmt=lambda v: f"[{float(v):.2f}]",
    )
    assert "[1.25]" in tex
    assert "[0.25]" in tex


def test_results_input_not_mutated():
    results = _toy_results()
    before_id = id(results[("a", "m1")])
    _ = dict_to_double_entry_latex_table(results)

    assert id(results[("a", "m1")]) == before_id
    assert isinstance(results[("a", "m1")], torch.Tensor)


def test_save_double_entry_table_writes_expected_file(tmp_path):
    path = save_double_entry_table(_toy_results(), plot_dir=str(tmp_path), suffix="toy")

    assert path == str(tmp_path / "toy_results.tex")
    assert Path(path).read_text(encoding="utf-8").startswith("% Requires")


def test_dict_to_double_entry_latex_table_rejects_invalid_metric_direction():
    with pytest.raises(ValueError, match="metric_direction"):
        dict_to_double_entry_latex_table(_toy_results(), metric_direction="sideways")


def test_dict_to_double_entry_latex_table_rejects_empty_results():
    with pytest.raises(ValueError, match="No columns"):
        dict_to_double_entry_latex_table({})


def test_dict_to_double_entry_latex_table_handles_zero_p_value(monkeypatch):
    monkeypatch.setattr("genkit.table.ttest_ind", lambda *args, **kwargs: (0.0, 0.0))

    tex = dict_to_double_entry_latex_table(_toy_results(), bold_best_in_row=True)

    assert r"{\scriptstyle 0}" in tex
