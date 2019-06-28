"""Load a casedir."""

import dolfin
import logging
import h5py

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

    def load_mesh(self) -> dolfin.mesh:
        """Load and return the mesh.

        Will also return cell and facet functions if present.
        """
        filename = self.casedir/Path("mesh.hdf5")
        mesh = dolfin.Mesh()
        with dolfin.HDF5File(mesh.mpi_comm(), str(filename), "r") as meshfile:
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
        mesh = self.load_mesh()

        element_tuple = (
            dolfin.interval,
            dolfin.triangle,
            dolfin.tetrahedron
        )

        element = dolfin.FiniteElement(
            metadata["element_family"],
            element_tuple[mesh.geometry().dim() - 1],        # zero indexed
            metadata["element_degree"]
        )

        V_space = dolfin.FunctionSpace(mesh, element)
        v_func = dolfin.Function(V_space)

        filename = self.casedir/Path("{name}/{name}.hdf5".format(name=name))
        with dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "r") as fieldfile:
            for i in _timestep_iterable:
                if i < int(metadata["start_timestep"]):
                    continue
                if i % int(metadata["stride_timestep"]) != 0:
                    continue
                # TODO: return function, not numpy array, and not petsc vector
                fieldfile.read(v_func, "{name}{i}".format(name=name, i=i))
                yield time_iterable[i], v_func

    def load_checkpoint(
            self,
            name: str,
            timestep_iterable: Iterable[int] = None,
    ) -> Iterator[Tuple[float, dolfin.Function]]:
        metadata = self.load_metadata(name)

        _timestep_iterable = timestep_iterable
        timestep_iterable, time_iterable = self.load_time()
        if _timestep_iterable is None:
            _timestep_iterable = timestep_iterable
        mesh = self.load_mesh()

        element_tuple = (
            dolfin.interval,
            dolfin.triangle,
            dolfin.tetrahedron
        )

        element = dolfin.FiniteElement(
            metadata["element_family"],
            element_tuple[mesh.geometry().dim() - 1],        # zero indexed
            metadata["element_degree"]
        )

        V_space = dolfin.FunctionSpace(mesh, element)
        v_func = dolfin.Function(V_space)

        filename = self.casedir/Path("{name}/{name}.hdf5".format(name=name))
        with dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "r") as fieldfile:
            for i, _time in enumerate(_timestep_iterable):
                if _time < int(metadata["start_timestep"]):
                    continue
                if _time % int(metadata["stride_timestep"]) != 0:
                    continue
                fieldfile.read_checkpoint(infunc, self.name, counter=i)
                yield time_iterable[i], v_func.vector()

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

    def load_initial_condition(
            self,
            name: str,
    ) -> Dict[str, dolfin.Function]:
        """Return the last computed values for the fields in `name_iterable`."""
        metadata = self.load_metadata(name)
        mesh = self.load_mesh()

        element_tuple = (
            dolfin.interval,
            dolfin.triangle,
            dolfin.tetrahedron
        )

        element = dolfin.FiniteElement(
            metadata["element_family"],
            element_tuple[mesh.geometry().dim() - 1],        # zero indexed
            metadata["element_degree"]
        )

        V_space = dolfin.FunctionSpace(mesh, element)
        v_func = dolfin.Function(V_space)

        filename = self.casedir/Path("{name}/{name}.hdf5".format(name=name))
        with h5py.File(filename, "r") as hdf5_file:
            sorted_field_names = sorted(hdf5_file.keys(), key=lambda x: int(x[1:]))

        with dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "r") as fieldfile:
            fieldfile.read(v_func, sorted_field_names[-1])

        timestep = metadata["start_timestep"] + metadata["stride_timestep"]*len(sorted_field_names)
        _, time = load_times(self.casedir)
        return v_func.vector().get_local(), time[timestep]
