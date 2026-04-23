"Latex table module."

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import ttest_ind
from .utils import to_numpy


def save_double_entry_table(
    results: Mapping[Tuple[Any, Any], Any],
    plot_dir: str,
    fmt: str | Callable[[Any], str] = "{:.4f}",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    suffix: str = "experiment",
    top_left: str = "",
    metric_direction: str | Mapping[Any, str] = "down",
    show_metric_arrows: bool = True,
    bold_best_in_row: bool = False,
    bold_ties: bool = True,
    bold_atol: float = 0.0,
    row_order: Optional[Sequence[Any]] = None,
    col_order: Optional[Sequence[Any]] = None,
    col_name: Optional[Mapping[str, str]] = None,
    metric_name: Optional[Mapping[str, str]] = None,
    col_align: str = "r",
    nan_str: str = "--",
) -> str:
    """Save `results[(approach, metric)] = value` as a LaTeX table (rows=metrics, cols=approaches)."""
    plot_dir_p = Path(plot_dir)
    plot_dir_p.mkdir(parents=True, exist_ok=True)
    tex_path = plot_dir_p / f"{suffix}_results.tex"

    tex = dict_to_double_entry_latex_table(
        results=results,
        fmt=fmt,
        caption=caption,
        label=label,
        top_left=top_left,
        metric_direction=metric_direction,
        show_metric_arrows=show_metric_arrows,
        bold_best_in_row=bold_best_in_row,
        bold_ties=bold_ties,
        bold_atol=bold_atol,
        row_order=row_order,
        col_order=col_order,
        col_name=col_name,
        metric_name=metric_name,
        col_align=col_align,
        nan_str=nan_str,
    )
    tex_path.write_text(tex, encoding="utf-8")
    return str(tex_path)


def dict_to_double_entry_latex_table(
    results: Mapping[Tuple[Any, Any], Any],
    fmt: str | Callable[[Any], str] = "{:.4f}",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    top_left: str = "",
    metric_direction: str | Mapping[Any, str] = "down",
    show_metric_arrows: bool = True,
    bold_best_in_row: bool = False,
    bold_ties: bool = True,
    bold_atol: float = 0.0,
    row_order: Optional[Sequence[Any]] = None,
    col_order: Optional[Sequence[Any]] = None,
    col_name: Optional[Mapping[str, str]] = None,
    metric_name: Optional[Mapping[str, str]] = None,
    col_align: str = "r",
    nan_str: str = "--",
) -> str:
    """Format `results[(approach, metric)]` into LaTeX; rows=metrics, cols=approaches; arrows by metric; optional bold best per row."""
    metrics = list(row_order) if row_order is not None else sorted({k[1] for k in results}, key=str)
    approaches = list(col_order) if col_order is not None else sorted({k[0] for k in results}, key=str)

    if not approaches:
        raise ValueError("No columns (approaches) inferred from `results`.")

    if isinstance(metric_direction, str):
        if metric_direction not in ["down", "up"]:
            raise ValueError(f"'metric_direction' must be in ['down', 'up'], got {metric_direction}")
        metric_direction = {m: metric_direction for m in metrics}

    for k, v in metric_direction.items():
        if v not in ['down', 'up']:
            raise ValueError(f"'metric_direction' for metric '{k}' not in ['down', 'up'], {v}")

    norm_results: dict[tuple[Any, Any], Any] = {}
    for m in metrics:
        for a in approaches:
            norm_results[(a, m)] = to_numpy(results[(a, m)])

    best_in_row: dict[Any, float] = {}
    best_name_in_row: dict[Any, float] = {}
    if bold_best_in_row:
        for m in metrics:

            vals = []
            for a in approaches:
                vals.append(np.mean(norm_results[(a, m)]))

            if vals:
                best = np.argmin(vals) if metric_direction[m] == "down" else np.argmax(vals)
                best_in_row[m] = vals[best]
                best_name_in_row[m] = approaches[best]

    def _fmt_ceil(x: float, t: float = 1e-1) -> str:
        """Format a p-value into a compact scientific-notation superscript string."""
        if (x < 0) or (x > t):
            return "-"
        if x == 0:
            return "0"
        return f"{np.sign(x) * 10:.0f}" + r"^{" + f"{np.ceil(np.log10(np.abs(x))):.0f}" + r"}"

    def _fmt_value(v: float) -> str:
        """Format a scalar table entry with either a callable or a format string."""
        if callable(fmt):
            return fmt(v)
        return fmt.format(v)

    def _is_best(v: Any, m: Any) -> bool:
        """Check whether a value should be highlighted as the best entry in its row."""
        if not bold_best_in_row or m not in best_in_row:
            return False
        fv = float(v)
        b = best_in_row[m]
        if bold_ties:
            return abs(fv - b) <= float(bold_atol)
        return fv == b

    header = top_left + " & " + " & ".join(map(str, [col_name[ap] if col_name else ap for ap in approaches])) + r" \\"

    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{" + "l" + (col_align * len(approaches)) + "}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
    ]

    for m in metrics:

        m_str = str(m)

        if metric_name:
            m_str = metric_name[m_str]

        if show_metric_arrows:
            m_str += " " + (r"$\downarrow$" if metric_direction[m] == "down" else r"$\uparrow$")

        cells = []
        for a in approaches:
            mean_v = np.mean(norm_results[(a, m)])
            std_v = np.std(norm_results[(a, m)])
            s_mean = _fmt_value(mean_v)
            s_std = _fmt_value(std_v)

            if _is_best(mean_v, m) and s_mean != nan_str and s_std != nan_str:
                s = r"\textbf{" + s_mean + r"$\pm$" + s_std + "}"
            else:
                if bold_best_in_row:
                    _, p = ttest_ind(
                        norm_results[(best_name_in_row[m], m)],
                        norm_results[(a, m)],
                        equal_var=False,
                        alternative='two-sided',
                        nan_policy='omit',
                    )
                    s = "$" + s_mean + r"\pm" + s_std + r"\;{\scriptstyle " + _fmt_ceil(p) + "}$"
                else:
                    s = "$" + s_mean + r"\pm" + s_std + "$"

            cells.append(s)

        lines.append(f"    {m_str} & " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    if caption is not None:
        lines.append(rf"\caption{{{caption}}}")

    if label is not None:
        lines.append(rf"\label{{{label}}}")

    lines.append(r"\end{table}")

    return "\n".join(lines)
