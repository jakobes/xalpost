"""Load a casedir."""
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
)


logger = logging.getLogger(__name__)


class Loader:
    """Read stuff."""

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

        element = eval(spec["element"])     # Let us hopw this does not go wring
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

    def get_time(self) -> np.ndarray:
        """Return the times."""
        filename = self.casedir/Path("times.npy")
        assert filename.isfile(), f"Cannot find {filename}"
        return np.load(filename)
