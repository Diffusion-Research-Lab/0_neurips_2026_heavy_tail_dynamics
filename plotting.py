"""Plotting functions for diffusion model experiments."""

# Authors: Hamza Cherkaoui

import os
from typing import Tuple, List
import matplotlib.pyplot as plt
from constants import FONTSIZE


def plot_training_loss(
    l_loss: List[float],
    plot_dir: str,
    figsize: Tuple = (5, 4),
    xlogscale: bool = False,
    ylogscale: bool = False,
    fontsize: int = FONTSIZE,
    suffix: str = "experiment",
) -> Tuple[str, str]:
    """Save training loss curve under plot_dir."""

    pdf_path = os.path.join(plot_dir, f"{suffix}_training_loss.pdf")
    epochs = list(range(1, len(l_loss) + 1))

    plt.figure(figsize=figsize)

    plt.plot(epochs, l_loss, linewidth=3.5, alpha=0.5, label="MSE train loss")

    if xlogscale:
        plt.xscale("log")
    if ylogscale:
        plt.yscale("log")
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(pdf_path, dpi=350)
    plt.close()

    return pdf_path
