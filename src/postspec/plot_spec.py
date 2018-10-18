"""Plot specifications."""

from typing import (
    NamedTuple,
    Tuple,
)


class PlotSpec(NamedTuple):
    outdir: str
    name: str
    title: str
    ylabel: str
    label_fs: int = 24      # fontsizes
    title_fs: int = 48
    xlabel: str = "Time (ms)"
    save_format: str = "png"
    label_loc: str = "best"
    figsize: Tuple[int, int] = (14, 14)
    grid: bool = True
    linewidth: int = 4
