from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_band_plot(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_mean, color="C0", lw=2)
    ax.fill_between(x, y_lo, y_hi, color="C0", alpha=0.25, linewidth=0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

