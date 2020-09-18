"""Load a casedir."""

import dolfin
import logging
# import h5py

import numpy as np
import dolfin as df

from postspec import (
    LoaderSpec,
    FieldSpec,
)

from postutils import load_metadata

from postfields import (
    Field,
)

from pathlib import Path

from typing import (
    Dict,
    Any,
    Iterable,
    Tuple,
    Iterator,
)

from .baseclass import PostProcessorBaseClass
from .load_plain_text import load_times


LOGGER = logging.getLogger(__name__)


class Loader(PostProcessorBaseClass):
    """Class for loading meshes and functions."""

    def __init__(self, spec: LoaderSpec) -> None:
        """Store saver specifications."""
        super().__init__(spec)
        self.mesh = None

    # TODO: @property?
    def set_mesh(self, mesh: df.Mesh) -> None:
        self.mesh = mesh

    def load_mesh(self) -> dolfin.mesh:
        """Load and return the mesh stored as xdmf."""
        if self.mesh is None:
            self.mesh = df.Mesh()
            mesh_name = self._casedir / Path("mesh.xdmf")
            with df.XDMFFile(str(mesh_name)) as infile:
                infile.read(self.mesh)
        return self.mesh

    def load_mesh_function(self, name: str) -> dolfin.MeshFunction:
        """Lead and return a mesh function.

        There are two options, 'cell_function' or 'facet_function'.

        Arguments:
            name: Either 'cell_function' or 'facet_function'.
        """
        # TODO: I could use Enum rather than hard-coding names
        msg = "Meshfunctions are stored as 'cell_function' or 'facet_function'."
        if not name in ("cell_function", "facet_function"):
            raise ValueError(msg)

        self.load_mesh()        # Method tests if mesh is already loaded

        dimension = self.mesh.geometry().dim()      # if cell function
        if name == "facet_function":
            dimension -= 1      # dimension is one less

        # mvc = df.MeshValueCollection("size_t", self.mesh, dimension)
        cell_function = df.MeshFunction("size_t", self.mesh, dimension)
        with df.XDMFFile(str(self._casedir / f"{name}.xdmf")) as infile:
            infile.read(cell_function)
            # infile.read(mvc)
        # cell_function = df.MeshFunction("size_t", self.mesh, mvc)
        return cell_function

    def load_metadata(self, name) -> Dict[str, str]:
        """Read the metadata associated with a field name."""
        return load_metadata(self._casedir / Path("{name}/metadata_{name}.yaml".format(name=name)))

    def load_field(
            self,
            name: str,
            timestep_iterable: Iterable[int] = None,
            vector: bool = False,
    ) -> Iterator[Tuple[float, dolfin.Function]]:
        """Return an iterator over the field for each timestep.

        TODO: Push this back to the specific field

        Optionally, return the corresponding time.
        """
        metadata = self.load_metadata(name)

        _timestep_iterable = timestep_iterable
        timestep_iterable, time_iterable = self.load_time()
        if _timestep_iterable is None:
            _timestep_iterable = timestep_iterable
        if self.mesh is None:
            self.mesh = self.load_mesh()

        element_tuple = (
            dolfin.interval,
            dolfin.triangle,
            dolfin.tetrahedron
        )

        if vector:
            element = dolfin.VectorElement(
                "CG",
                element_tuple[self.mesh.geometry().dim() - 1],
                1
            )
        else:
            element = dolfin.FiniteElement(
                "CG",
                element_tuple[self.mesh.geometry().dim() - 1],        # zero indexed
                1
            )

        V_space = dolfin.FunctionSpace(self.mesh, element)
        v_func = dolfin.Function(V_space)

        filename = self._casedir / name / f"{name}.hdf5"
        with dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "r") as fieldfile:
        # with dolfin.XDMFFile(self.mesh.mpi_comm(), str(filename)) as fieldfile:
            for savad_timestep_index, timestep in enumerate(_timestep_iterable):
                if timestep < int(metadata["start_timestep"]):
                    continue
                if timestep % int(metadata["stride_timestep"]) != 0:
                    continue
                # assert False, (v_func, f"{name}", i)
                # fieldfile.read(v_func, f"{name}{i}/vector")
                fieldfile.read(v_func, f"{name}{timestep}")
                yield time_iterable[savad_timestep_index], v_func

    def load_checkpoint(
        self,
        name: str,
        timestep_iterable: Iterable[int] = None,
    ) -> Iterator[Tuple[int, float, dolfin.Function]]:
        """yield tuple(float, function)."""
        metadata = self.load_metadata(name)

        _timestep_iterable = timestep_iterable
        timestep_iterable, time_iterable = self.load_time()
        if _timestep_iterable is None:
            _timestep_iterable = timestep_iterable

        if self.mesh is None:
            self.mesh = self.load_mesh()

        element_tuple = (
            dolfin.interval,
            dolfin.triangle,
            dolfin.tetrahedron
        )

        element = dolfin.FiniteElement(
            metadata["element_family"],
            element_tuple[self.mesh.geometry().dim() - 1],        # zero indexed
            metadata["element_degree"]
        )

        V_space = dolfin.FunctionSpace(self.mesh, element)
        v_func = dolfin.Function(V_space)

        _filename = self.casedir / Path("{name}/{name}_chk.xdmf".format(name=name))
        if _filename.exists():
            filename_list = [_filename]
        else:
            filename_list = []
            filename_directory = self.casedir / Path(f"{name}")
            for _file in filter(lambda x: x.suffix == ".xdmf", filename_directory.iterdir()):
                if "_chk_part" in _file.stem:
                    LOGGER.info(f"loading file {_file}")
                    filename_list.append(_file)

        for filename in filename_list:
            with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as fieldfile:
                for savad_timestep_index, timestep in enumerate(_timestep_iterable):
                    if timestep < int(metadata["start_timestep"]):
                        continue
                    if timestep % int(metadata["stride_timestep"]) != 0:
                        continue
                    try:
                        fieldfile.read_checkpoint(v_func, name, counter=timestep)
                    except RuntimeError as e:
                        LOGGER.info(f"Could not read timestep: {e}")
                    yield time_iterable[savad_timestep_index], v_func

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir

    def load_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (timesteps, times)."""
        # filename = self.casedir / Path("times.txt")
        filename = self.casedir
        assert filename.exists(), "Cannot find {filename}".format(filename=filename)
        return load_times(filename)
