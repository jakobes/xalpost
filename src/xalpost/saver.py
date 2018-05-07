"""An interface for saving a `Field` as hdf5."""

import dolfin
import logging
import yaml

from .baseclass import PostProcessorBaseClass

from .field import Field

from postspec import (
    PostProcessorSpec,
    FieldSpec,
)

from pathlib import Path

from typing import (
    Dict,
)


LOGGER = logging.getLogger(__name__)


class Saver(PostProcessorBaseClass):
    """Class for saving stuff."""

    def __init__(self, spec: PostProcessorSpec) -> None:
        """Store saver specifications."""
        super().__init__()
        self._time_list = []            # Keep track of time points
        self._first_compute = True      # Perform special action after before first save
        self._fields = {}               # Dict tocheck for duplicates

    def store_mesh(
            self,
            mesh: dolfin.Mesh,
            cell_domain: dolfin.MeshFunction = None,
            facet_domain: dolfin.MeshFunction = None
    ) -> None:
        """Save the mesh, and cellfunction and facet function if provided."""
        filename = casedir/Path("mesh.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), filename, "w") as meshfile:
            meshfile.write(mesh, "Mesh")
            if cell_domains is not None:
                meshfile.write(cell_domains, "CellDomains")
            if facet_domains is not None:
                meshfile.write(facet_domains, "FacetDomains")

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
            new_data: Dict[str, dolfin.Function]
    ) -> None:
        """Store solutions and perform computations for new timestep."""
        for field_name, data in new_data.items():
            field = self._field_dict[field_name]
            field.update(time, timestep, data)

        for name in field_dict:
            spec = self._fields[name].spec
            if spec.stride_timestep % int(timestep) and sepc.start_timestep >= timestep:
                if field.first_compute:     # Store metadata if not already done
                    spec_dict = spec._asdict() 
                    element = str(field_dict[name].function_space().ufl_element())
                    spec_dict["element"] = element
                    name = f"{name}/{name}"
                    self.store_metadata(name, spec_dict)
                    field.first_compute = False

                self.store_field(
                    field_dict[name],
                    timestep,
                )

        self._time_list.append(time)

    def finalise(self) -> None:
        """Store the times."""
        filename = self.casedir/Path("times.npy")
        np.save(filename, np.asarray(self._time_list)) 
