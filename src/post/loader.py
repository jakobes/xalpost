"""Load a casedir."""

import dolfin
import logging

import numpy as np

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
)

from .baseclass import PostProcessorBaseClass
from .load_plain_text import load_times


LOGGER = logging.getLogger(__name__)


class Loader(PostProcessorBaseClass):
    """Class for loading meshes and functions."""

    def __init__(self, spec: LoaderSpec) -> None:
        """Store saver specifications."""
        super().__init__(spec)

    def load_mesh(self) -> dolfin.mesh:
        """Load and return the mesh.

        Will also return cell and facet functions if present.
        """
        filename = self.casedir/Path("mesh.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), str(filename), "r") as meshfile:
            mesh = dolfin.Mesh()
            meshfile.read(mesh, "/Mesh", False)
        return mesh

    def load_mesh_function(self, mesh: dolfin.Mesh, name: str) -> dolfin.MeshFunction:
        """Lead and return a mesh function.

        There are two options, 'CellDomains' or 'FacetDomains'. Both are stored in
        'mesh.hdf5'.

        Arguments:
            mesh: The mesh the function is defin on. Use `self.load_mesh`.
            name: Either 'CellDomains' or 'FacetDomains'.
        """
        msg = "Meshfunctions are stored as 'CellDomains' or 'FacetDomains'."
        assert name in ("CellDomains", "FacetDomains"), msg

        filename = self.casedir/Path("mesh.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), str(filename), "r") as meshfile:
            mesh_function = dolfin.MeshFunction("size_t", mesh)
            meshfile.read(mesh_function, "/{name}".format(name=name))
        return mesh_function

    def load_metadata(self, name) -> Dict[str, str]:
        """Read the metadata associated with a field name."""
        return load_metadata(self.casedir/Path("{name}/metadata_{name}.yaml".format(name=name)))

    def load_field(
            self,
            name: str,
            timestep_iterable: Iterable[int] = None,
            return_time = True
    ) -> Any:       # FIXME: return type
        """Return an iterator over the field for each timestep.

        Optionally, return the corresponding time.
        """
        metadata = self.load_metadata(name)

        if timestep_iterable is None or return_time:
            timestep_iterable, time_iterable = self.load_time()
        mesh = self.load_mesh()

        element = dolfin.FiniteElement(
            metadata["element_family"],
            dolfin.triangle,
            metadata["element_degree"]
        )
        V_space = dolfin.FunctionSpace(mesh, element)
        v_func = dolfin.Function(V_space)

        filename = self.casedir/Path("{name}/{name}.hdf5".format(name=name))
        with dolfin.HDF5File(dolfin.mpi_comm_world(), str(filename), "r") as fieldfile:
            for i in timestep_iterable:
                if i < int(metadata["start_timestep"]):
                    continue
                if i % int(metadata["stride_timestep"]) != 0:
                    continue

                fieldfile.read(v_func, "{name}{i}".format(name=name, i=i))
                if return_time:
                    yield v_func, time_iterable[i]
                else:
                    yield v_func

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir

    def load_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (timesteps, times)."""
        filename = self.casedir / Path("times.txt")
        assert filename.exists(), "Cannot find {filename}".format(filename)
        return load_times(filename)

    def load_initial_condition(
            self,
            name: str,
            timestep: int = None
    ) -> Dict[str, dolfin.Function]:
        """Return the last computed values for the fields in `name_iterable`."""

        if timestep is None:
            timestep, _ = self.load_time()[-1]

        field = next(self.load_field(name, (timestep,), return_time=False))
        return field
