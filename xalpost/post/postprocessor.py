"""A postprocessor for saving and loading a mesh, meshfunctions and functions for hdf5."""

import dolfin
import logging
import yaml

from xalpost.spec import (
    PostProcessorSpec,
    FieldSpec,
)

from xalpost.post import (
    Field,
)

from pathlib import Path

from typing import (
    Dict,
    Namedtuple,
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

    def load_metadata(self, name) -> Dict[str, str]:
        """Read the metadata associated with a field name."""
        field = self._fields[name]
        with open(self.casedir/Path(f"{name}/{name}.yaml"), "r") as in_handle:
            return yaml.load(in_handle)

    def load_field(self, name: str) -> None:
        """Return an iterator over the field for each timestep."""
        field = self._fields[name]
        metadata = self.load_metadata(name)

        time_array = self.get_time()
        mesh = self.load_mesh()

        element = dolfin.FiniteElement(     # FIXME: What about vector elements?
            metadata["element_family"],
            metadata["element_cell"],
            metadata["element_degree"]
        )
        V = dolfin.FunctionSpace(mesh, element)
        v = Function(V)

        filename = self.casedir/Path(f"{name}/{name}.hdf5")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), filename, "r") as fieldfile:
            for i, t in enumerate(time_array):
                fieldfile.read(v, "/{name}{i}")
                yield t, v

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir

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

    def get_time(self) -> np.ndarray:
        """Return the times."""
        filename = self.casedir/Path("times.npy")
        assert filename.isfile(), f"Cannot find {filename}"
        return np.load(filename)

