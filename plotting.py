"""Plotting functions for diffusion model experiments."""

# Authors: Hamza Cherkaoui

import os
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from constants import FONTSIZE
from utils import format_tick


def plot_training_loss(
    l_loss: List[float | torch.Tensor],
    plot_dir: str | os.PathLike,
    figsize: Tuple = (5, 4),
    xlogscale: bool = False,
    ylogscale: bool = False,
    fontsize: int = FONTSIZE,
    suffix: str = "experiment",
) -> Tuple[str, str]:
    """Save training loss curve under plot_dir."""

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"{suffix}_training_loss.pdf"

    epochs = list(range(1, len(l_loss) + 1))

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
    plot_dir: str | os.PathLike,
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
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu().item())
        clean_results[(float(nu_p1), float(nu_p0))] = float(v)

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
        im = ax.matshow(Z, origin="lower", aspect="auto", norm=norm)
    else:
        im = ax.matshow(Z, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n0))
    ax.set_yticks(np.arange(n1))
    ax.set_xticklabels([format_tick(v) for v in xticks], fontsize=fontsize, rotation=45)
    ax.set_yticklabels([format_tick(v) for v in yticks], fontsize=fontsize, rotation=45)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return str(pdf_path)
