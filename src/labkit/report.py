"""Reporting helpers for tables, figures, and LaTeX-ready strings."""

import numpy as np

PRETTY_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.9,
    "axes.axisbelow": True,
    "lines.linewidth": 1.75,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
}


def _to_latex_sci(x: float, digits: int = 2) -> str:
    """Format a scalar in compact LaTeX scientific notation."""
    if not np.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10**exponent)
    if exponent == 0:
        return f"{x:.{digits}f}"
    return rf"{mantissa:.{digits}f}\,10^{{{exponent}}}"


def format_mean_std_latex(mean: float, std: float, caption: str = "", bold: bool = False) -> str:
    """Format a mean-plus-std pair as a LaTeX mathtext string."""
    mean_str = _to_latex_sci(mean, digits=2)
    core = rf"\mathbf{{{mean_str}}}" if bold else mean_str
    if not np.isclose(std, 0.0):
        std_str = _to_latex_sci(std, digits=2)
        core = rf"{core}_{{\pm {std_str}}}"
    if caption:
        core = rf"\underset{{\mathrm{{{caption}}}}}{{{core}}}"
    return rf"${core}$"
