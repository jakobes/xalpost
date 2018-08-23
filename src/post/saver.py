"""An interface for saving a `Field` as hdf5."""

import dolfin
import logging

import numpy as np

from postspec import (
    PostProcessorSpec,
    SaverSpec,
    LoaderSpec,
    FieldSpec,
)

from postfields import (
    Field,
)

from pathlib import Path

from typing import (
    Dict,
    Any,
    List,
)

from .baseclass import PostProcessorBaseClass


LOGGER = logging.getLogger(__name__)


class Saver(PostProcessorBaseClass):
    """Class for saving stuff."""

    def __init__(self, spec: LoaderSpec) -> None:
        """Store saver specifications."""
        super().__init__(spec)
        self._time_list: List[float] = []            # Keep track of time points
        self._first_compute = True      # Perform special action after before first save

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

    def add_field(self, field: Field) -> None:
        """Add a field to the postprocessor."""
        # FIXME: Figure out the logging system
        msg = f"A field with name {field.name} already exists."
        assert field.name not in self._fields, msg      # TODO: Issue warning, not abort
        field.path = self._casedir
        self._fields[field.name] = field

    def update(
            self,
            time: float,
            timestep: int,
            data_dict: Dict[str, dolfin.Function]
    ) -> None:
        """Store solutions and perform computations for new timestep."""
        self._time_list.append(float(time))    # This time array has to be sent to each field
        for name, data in data_dict.items():
            self._fields[name].update(timestep, time, data)

    def close(self) -> None:
        """Store the times."""
        filename = self.casedir/Path("times.npy")
        print(np.asarray(self._time_list))
        np.save(filename, np.asarray(self._time_list))
        for _, field in self._fields.items():
            field.close()
