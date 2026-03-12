"""Table module unittests."""

import torch
from cauda.table import dict_to_double_entry_latex_table


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
