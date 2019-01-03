"""Plot PointField over time."""

from pathlib import Path
from typing import Container

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from postspec import PlotSpec


def plot_point_field(
        times: Container[float],
        probes: np.ndarray,
        spec: PlotSpec,
        labels: Container[str] = None) -> None:
    """Plot the probes from `PointField`."""
    # TODO: Might need to update in light of probes plot
    fig = plt.figure(figsize=spec.figsize)
    ax = fig.add_subplot(111)

    for p in probes:
        ax.plot(times, p, linewidth=spec.linewidth)

    ax.set_title(spec.title, fontsize=spec.title_fs)
    ax.set_xlabel(spec.xlabel, fontsize=spec.label_fs)
    ax.set_ylabel(spec.ylabel, fontsize=spec.label_fs)
    ax.grid(spec.grid)

    if labels is None:
        labels = [f"probe {i}" for i in range(len(probes))]
    ax.legend(labels, loc=spec.label_loc)

    outdir = Path(spec.outdir)
    outdir.mkdir(exist_ok=True)
    fig.savefig(str(outdir / f"{spec.name}.{spec.save_format}"))
    plt.close(fig)
