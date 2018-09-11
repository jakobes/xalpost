from typing import (
    NamedTuple,
    Tuple,
)


try:
    class SaverSpec(NamedTuple):
        """Specifications for `post.Saver`."""
        casedir: str
except SyntaxError:
    class SaverSpec:
        def __init__(self, casedir):
            self.casedir = casedir


try:
    class LoaderSpec(NamedTuple):
        """Specifications for `post.Saver`."""
        casedir: str
except SyntaxError:
    class SaverSpec:
        def __init__(self, casedir):
            self.casedir = casedir


try:
    class PostProcessorSpec(NamedTuple):
        """Specifications for `PostProcessor`."""
        casedir: str    # Name of output directory
except SyntaxError:
    class PostProcessorSpec:
        def __init__(self, casedir):
            self.casedir = casedir

try:
    class FieldSpec:
        def __init__(
                self,
                save,
                save_as=("hdf5", "xdmf"),
                plot=False,
                start_timestep=-1,
                stride_timestep=1,
                element_family=None,
                element_degree=None
        ):
            self.save = save
            self.save_as = save_as
            self.plot = plot
            self.start_timestep = start_timestep
            self.stride_timestep = stride_timestep
            self.element_family = element_family
            self.element_degree = element_degree
except SyntaxError:
    class FieldSpec(NamedTuple):
        save: bool = True
        save_as: Tuple[str] = ("hdf5", "xdmf")
        plot: bool = False
        start_timestep: int = -1    # Save after `start_timestep` timestep
        stride_timestep: int = 1    # Save every `stride_timestep`
        element_family: str = None
        element_degree: int = None
