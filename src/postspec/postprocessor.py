from pathlib import Path

from typing import (
    NamedTuple,
    Tuple,
)

import typing as tp


class SaverSpec(NamedTuple):
    """Specifications for `post.Saver`.
    """
    casedir: Path
    overwrite_casedir: bool = False


class LoaderSpec(NamedTuple):
    """Specifications for `post.Saver`."""
    casedir: Path


class PostProcessorSpec(NamedTuple):
    """Specifications for `PostProcessor`."""
    casedir: str    # Name of output directory


class FieldSpec(NamedTuple):
    """
    `num_steps_in_part` is the number of timesteps store before a output file is broken into parts.
    """
    save: bool = True
    save_as: Tuple[str] = ("checkpoint",)
    plot: bool = False
    start_timestep: int = -1    # Save after `start_timestep` timestep
    stride_timestep: int = 1    # Save every `stride_timestep`
    element_family: tp.Optional[str] = None
    element_degree: tp.Optional[int] = None
    sub_field_index: tp.Optional[int] = None     # The index of the subfunction space
    num_steps_in_part: tp.Optional[int] = None
