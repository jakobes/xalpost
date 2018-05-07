from typing import (
    NamedTuple,
    Tuple,
)


class PostProcessorSpec(NamedTuple):
    """Specifications for `PostProcessor`."""
    casedir: str    # Name of output directory


class FieldSpec(NamedTuple):
    save: bool = True
    save_as: Tuple[str] = ("hdf5", "xdmf")
    plot: bool = False
    start_timestep: int = -1    # Save after `start_timestep` timestep
    stride_timestep: int = 1    # Save every `stride_timestep`
    element_family: str = None
    element_cell: str = None
    element_fdegree: int = None