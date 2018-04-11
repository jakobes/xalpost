"""An interface for saving a `Field` as hdf5."""

import dolfin
import logging
import yaml

from .baseclass import PostProcessorBaseClass

from xalpost.spec import (
    PostProcessorSpec,
    FieldSpec,
)

from pathlib import Path

from typing import (
    Dict,
)


logger  = logging.getLogger(__name__)


class Saver(PostProcessorBaseClass):
    """Class for saving stuff."""

    def __init__(self, spec: PostProcessorSpec) -> None:
        """Store saver specifications."""
        super().__init__()
        self._time_list = []            # Keep track of time points
        self._first_compute = True      # Perform special action after before first save

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

    def store_field(self, function: dolfin.Function, timestep: int) -> None:
        """Save the function, and cellfunction and facet function if provided."""
        filename = casedir/Path("{name}/{name}.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), filename, "w") as fieldfile:
            if not datafile.has_dataset("Mesh"):
                fieldfile.write(data.function_space().mesh(), "Mesh")
                datafile.write(data, f"{name}{timestep}")

    def store_metadata(
            self,
            name: str,
            spec: Dict[Any Any],
            default_flow_style: bool = False
    ) -> None:
        """Save spec as {name}.yaml.

        `name` is converted to a `Path` and save relative to `self.casedir`.

        Arguments:
            name: Name of yaml file.
            spec: Anything compatible with pyaml. It is converted to yaml and dumped.
            default_flow_style: use default_flow_style.
        """
        filename = self.casedir/Path(f"{name}.yaml")
        with open(filename, "w") as out_handle:
            yaml.dump(spec, out_handle, default_flow_style=default_flow_style)

    def add_field(self, field: Field) -> None:
        """Add a field to the postprocessor."""
        msg = f"A field with name {field.name} already exists."
        assert field.name not in self._fields, msg 
        self._fields[field.name] = field

    def update(
            self,
            field_dict[str, dolfin.Function],
            time: float,
            timestep: int
    ) -> None:
        """Store solutions and perform computations for new timestep."""
        for name in field_dict:
            spec = self._fields[name].spec
            if spec.stride_timestep % int(timestep) and sepc.start_timestep >= timestep:
                if field.first_compute:     # Store metadata if not already done
                    spec_dict = spec._asdict() 
                    element = field_dict[name].function_space().ufl_element()
                    spec_dict["element_family"] = str(element.family())
                    spec_dict["element_cell"] = str(element.cell())
                    spec_dict["element_degree"] = str(element.degree())
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
