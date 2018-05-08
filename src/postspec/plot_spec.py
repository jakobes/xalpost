from typing import (
    NamedTuple,
    Tuple,
)


class PlotSpec(NamedTuple):
    outdir: str
    name: str
    save_format: str = "png"
    title: str
    xlabel: str = "Time (ms)"
    ylabel: str
    figsize: Tuple[int, int] = (14, 14)
    grid: bool = True
    line_width: int = 4
    label_loc: str = "best"
