"""A postprocessor for saving and loading a mesh, meshfunctions and functions for hdf5."""

import dolfin
import logging

import numpy as np

from postspec import (
    PostProcessorSpec,
    FieldSpec,
)

from xalpost import (
    Field,
)

from pathlib import Path

from typing import (
    Dict,
    Any,
)


LOGGER = logging.getLogger(__name__)


class PostProcessor:
    """Class for file I/O."""

    def __init__(self, spec: PostProcessorSpec) -> None:
        """Stor and process specifications."""
        self.spec = spec
        self._casedir = Path(spec.casedir)
        self._fields = {}
        self._time_list = []
        self._first_compute = True

    def store_mesh(
            self,
            mesh: dolfin.Mesh,
            cell_domains: dolfin.MeshFunction = None,
            facet_domains: dolfin.MeshFunction = None
    ) -> None:
        """Save the mesh, and cellfunction and facet function if provided."""
        filename = self.casedir/Path("mesh.hdf5")
        with dolfin.HDF5File(mesh.mpi_comm(), str(filename), "w") as meshfile:
            meshfile.write(mesh, "/Mesh")
            if cell_domains is not None:
                meshfile.write(cell_domains, "/CellDomains")
            if facet_domains is not None:
                meshfile.write(facet_domains, "/FacetDomains")

    def load_mesh(self) -> dolfin.mesh:
        """Load and return the mesh.

        Will also return cell and facet functions if present.
        """
        filename = calsedir/Path("mesh.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), filename, "r") as meshfile:
            mesh = dolfin.Mesh()
            meshfile.read(mesh, "/Mesh", False)
        return mesh

    def load_mesh_function(self, name: str) -> dolfin.MeshFunction:
        """Lead and return a mesh function.

        There are two options, 'CellDomains' or 'FacetDomains'. Both are stored in
        'mesh.hdf5'.

        Arguments:
            name: Either 'CellDomains' or 'FacetDomains'.
        """
        msg = "Meshfunctions are stored as 'CellDomains' or 'FacetDomains'."
        assert name in ("CellDomains", "FacetDomains"), msg

        filename = calsedir/Path("mesh.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), filename, "r") as meshfile:
            mesh_function = dolfin.MeshFunction()
            meshfile.read(mesh, f"/{name}")
        return mesh_function

    def add_field(self, field: Field) -> None:
        """Add a field to the postprocessor."""
        msg = f"A field with name {field.name} already exists."
        assert field.name not in self._fields, msg 
        field.path = self._casedir
        self._fields[field.name] = field

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir

    def update(
            self,
            time: float,
            timestep: int,
            data_dict: Dict[str, dolfin.Function]
    ) -> None:
        """Store solutions and perform computations for new timestep."""
        self._time_list.append(time)    # This time array has to be sent to each field
        for name, data in data_dict.items():
            self._fields[name].update(timestep, time, data)

    def finalise(self) -> None:
        """Store the times."""
        filename = self.casedir/Path("times.npfunctiony")
        np.save(filename, np.asarray(self._time_list)) 
        for _, field in self._fields.items():
            field.finalise()


    def get_time(self) -> np.ndarray:
        """Return the times."""
        filename = self.casedir/Path("times.npy")
        assert filename.isfile(), f"Cannot find {filename}"
        return np.load(filename)
