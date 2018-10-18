"""Load a casedir."""

import dolfin
import logging

import numpy as np

from postspec import (
    LoaderSpec,
    PostProcessorSpec,
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
)

from .baseclass import PostProcessorBaseClass
from .load_plain_text import load_times


LOGGER = logging.getLogger(__name__)


class Loader(PostProcessorBaseClass):
    """Class for loading meshes and functions."""

    def __init__(self, spec: LoaderSpec) -> None:
        """Store saver specifications."""
        super().__init__(spec)
        # self._time_list: List[float] = []            # Keep track of time points
        # self._first_compute = True      # Perform special action after before first save

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

    def load_field(self, name: str) -> None:
        """Return an iterator over the field for each timestep."""
        metadata = self.load_metadata(name)

        time_array = self.load_time()
        mesh = self.load_mesh()

        # element = eval(spec["element"])     # Let us hopw this does not go wring
        element = dolfin.FiniteElement(
            metadata["element_family"],
            dolfin.triangle,
            metadata["element_degree"]
        )
        V_space = dolfin.FunctionSpace(mesh, element)
        v_func = dolfin.Function(V_space)

        filename = self.casedir/Path("{name}/{name}.hdf5".format(name=name))
        with dolfin.HDF5File(dolfin.mpi_comm_world(), str(filename), "r") as fieldfile:
            for i, t in enumerate(time_array):
                if i < int(metadata["start_timestep"]):
                    continue
                if i % int(metadata["stride_timestep"]) != 0:
                    continue

                fieldfile.read(v_func, "{name}{i}".format(name=name, i=i))
                yield t, v_func

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir

    def load_time(self) -> np.ndarray:
        """Return the times."""
        filename = self.casedir / Path("times.txt")
        assert filename.exists(), "Cannot find {filename}".format(filename)
        return load_times(filename)
        # return np.load(filename)
