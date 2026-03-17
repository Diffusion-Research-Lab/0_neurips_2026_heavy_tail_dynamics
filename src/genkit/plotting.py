"""Plotting functions for diffusion model experiments."""

from pathlib import Path
from typing import Tuple, List, Dict, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from sklearn.decomposition import PCA
from .utils import to_numpy


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


def _format_tick(
    v: float,
    dec: int = 2,
    tol: float = 1e-6,
) -> str:
    """Format tick value: if integer-valued, format as int, else 3 significant digits."""
    if np.isfinite(v) and abs(v - round(v)) < tol:
        return str(int(round(v)))
    return f"{v:.{dec}g}"


def plot_scatter(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    plot_dir: str,
    perc_to_plot: float = 0.99,
    pad: float = 1.05,
    figsize: Tuple = (5, 4),
    fontsize: int = 18,
    suffix: str = "experiment",
) -> str:
    """Save 2d-scatter under plot_dir."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"{suffix}_2d_scatter.pdf"

    x = to_numpy(x)
    x_ref = to_numpy(x_ref)

    center = np.median(np.vstack([x, x_ref]), axis=0)
    half_xy = np.quantile(np.abs(np.vstack([x, x_ref]) - center), perc_to_plot, axis=0)
    half = float(np.max(half_xy))

    plt.figure(figsize=figsize)

    plt.scatter(x_ref[:, 0], x_ref[:, 1], s=6, label="Target", alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], s=6, label="Generated", alpha=0.25)

    plt.xlim(center[0] - pad * half, center[0] + pad * half)
    plt.ylim(center[1] - pad * half, center[1] + pad * half)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(fontsize=fontsize)

    plt.tight_layout(pad=0.8)
    plt.savefig(pdf_path, dpi=300)
    plt.close()

    return str(pdf_path)


def plot_histogram(
    x: Union[torch.Tensor, np.ndarray],
    x_ref: Union[torch.Tensor, np.ndarray],
    plot_dir: str,
    figsize: Tuple[int, int] = (5, 4),
    bins: int = 60,
    density: bool = True,
    fontsize: int = 18,
    suffix: str = "experiment",
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> str:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"{suffix}_hist.pdf"

    x = to_numpy(x)
    x_ref = to_numpy(x_ref)

    pca = PCA(n_components=1).fit(np.concatenate([x_ref, x], axis=0))
    x_1d = pca.transform(x).ravel()
    x_ref_1d = pca.transform(x_ref).ravel()

    q_lo, q_hi = clip_quantiles
    lo = float(np.quantile(np.concatenate([x_1d, x_ref_1d]), q_lo))
    hi = float(np.quantile(np.concatenate([x_1d, x_ref_1d]), q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(np.concatenate([x_1d, x_ref_1d])))
        hi = float(np.max(np.concatenate([x_1d, x_ref_1d])))

    plt.figure(figsize=figsize)

    plt.hist(x_ref_1d, bins=bins, density=density, alpha=0.5, label="Target", range=(lo, hi))
    plt.hist(x_1d, bins=bins, density=density, alpha=0.25, label="Generated", range=(lo, hi))

    plt.xlim(lo, hi)
    plt.legend(fontsize=fontsize)

    plt.tight_layout(pad=0.8)
    plt.savefig(pdf_path, dpi=350)
    plt.close()

    return str(pdf_path)


def plot_training_loss(
    l_loss: List[float | torch.Tensor],
    plot_dir: str,
    figsize: Tuple = (5, 4),
    xlogscale: bool = False,
    ylogscale: bool = False,
    fontsize: int = 18,
    suffix: str = "experiment",
) -> str:
    """Save training loss curve under plot_dir."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"{suffix}_training_loss.pdf"

    epochs = list(range(1, len(l_loss) + 1))
    l_loss = to_numpy(l_loss)

    plt.figure(figsize=figsize)

    plt.plot(epochs, l_loss, linewidth=3.5, alpha=0.5, label="MSE train loss")
    if xlogscale:
        plt.xscale("log")
    if ylogscale:
        plt.yscale("log")
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.tight_layout(pad=0.8)
    plt.savefig(pdf_path, dpi=350)
    plt.close()

    return str(pdf_path)


def plot_heatmap(
    results: Dict[Tuple[float, float], float | torch.Tensor],
    plot_dir: str,
    xlabel: str = "x-axis",
    ylabel: str = "y-axis",
    fontsize: int = 12,
    suffix: str = "flowmatching_longtail",
    clip_percentiles: tuple[float, float] | None = (5.0, 95.0),
    log_scale: bool = False,
) -> str:
    """Plot a heatmap of metric values stored in `results` keyed by (nu_p1, nu_p0). """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"heatmap_metric_{suffix}.pdf"

    clean_results: Dict[Tuple[float, float], float] = {}
    for (nu_p1, nu_p0), v in results.items():
        clean_results[(float(nu_p1), float(nu_p0))] = np.mean(to_numpy(v))

    yticks = sorted({k[0] for k in clean_results.keys()})
    xticks = sorted({k[1] for k in clean_results.keys()})

    n1 = len(yticks)
    n0 = len(xticks)

    Z = np.full((n1, n0), np.nan, dtype=float)
    for i, y in enumerate(yticks):
        for j, x in enumerate(xticks):
            if (y, x) in clean_results:
                Z[i, j] = clean_results[(y, x)]

    vals = Z[np.isfinite(Z)]

    vmin = vmax = None
    norm = None

    if log_scale:
        vals_pos = vals[vals > 0]
        if vals_pos.size == 0:
            raise ValueError("No positive finite values to plot on a log scale.")

        if clip_percentiles is not None:
            lo, hi = clip_percentiles
            vmin, vmax = np.percentile(vals_pos, [lo, hi])
        else:
            vmin, vmax = vals_pos.min(), vals_pos.max()

        eps = 1e-12
        vmin = max(vmin, eps)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin = vmax = None
    else:
        if clip_percentiles is not None and vals.size > 0:
            lo, hi = clip_percentiles
            vmin, vmax = np.percentile(vals, [lo, hi])

    fig, ax = plt.subplots(figsize=(max(5.5, 0.65 * n0), max(4.0, 0.6 * n1)))

    if norm is not None:
        im = ax.imshow(Z, origin="lower", aspect="auto", norm=norm, interpolation='gaussian')
    else:
        im = ax.imshow(Z, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, interpolation='gaussian')

    xticks_ = [0, int(n0 / 2), n0 - 1]
    yticks_ = [0, int(n1 / 2), n1 - 1]
    xticklabels_ = [xticks[0], xticks[int(n0 / 2)], xticks[n0 - 1]]
    yticklabels_ = [yticks[0], yticks[int(n1 / 2)], yticks[n1 - 1]]

    ax.set_xticks(xticks_)
    ax.set_yticks(yticks_)
    ax.set_xticklabels([_format_tick(v) for v in xticklabels_], fontsize=fontsize, rotation=45)
    ax.set_yticklabels([_format_tick(v) for v in yticklabels_], fontsize=fontsize, rotation=45)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return str(pdf_path)
