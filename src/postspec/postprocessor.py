from pathlib import Path

from typing import (
    NamedTuple,
    Tuple,
)


class SaverSpec(NamedTuple):
    """Specifications for `post.Saver`."""
    casedir: Path
    overwrite_casedir: bool = False


class LoaderSpec(NamedTuple):
    """Specifications for `post.Saver`."""
    casedir: Path


class PostProcessorSpec(NamedTuple):
    """Specifications for `PostProcessor`."""
    casedir: str    # Name of output directory


class FieldSpec(NamedTuple):
    save: bool = True
    save_as: Tuple[str] = ("checkpoint",)
    plot: bool = False
    start_timestep: int = -1    # Save after `start_timestep` timestep
    stride_timestep: int = 1    # Save every `stride_timestep`
    element_family: str = None
    element_degree: int = None
    sub_field_index: int = None     # The index of the subfunction space
