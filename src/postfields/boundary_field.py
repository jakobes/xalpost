import dolfin as df
from .field import Field

from postspec import FieldSpec

from postutils import (
    store_metadata,
)


class BoundaryField(Field):
    def __init__(self, name: str, spec: FieldSpec, mesh: df.Mesh):

        self._boundary_mesh = df.BoundaryMesh(mesh, "exterior")
        self._boundary_function_space = df.FunctionSpace(
            self._boundary_mesh,
            "CG",
            1
        )
        self._data = df.Function(self._boundary_function_space)
        super().__init__(name, spec)

    def _save_bmesh(self):
        mesh_path = self.path / "boundary_mesh.xdmf"
        with df.XDMFFile(str(mesh_path)) as mesh_file:
            mesh_file.write(self._boundary_mesh)

    def update(self, timestep: int, time: float, data: df.Function) -> None:
        """Update the data."""
        if not self.save_this_timestep(timestep, time):
            return

        if self._first_compute:
            self._first_compute = False
            if df.MPI.rank(df.MPI.comm_world) == 0:
                self._path.mkdir(parents=False, exist_ok=True)
            df.MPI.barrier(df.MPI.comm_world)

            # Update spec with element specifications
            spec_dict = self.spec._asdict()
            element = data.function_space().ufl_element()
            spec_dict["element_family"] = str(element.family())  # e.g. Lagrange
            spec_dict["element_degree"] = element.degree()

            self._save_bmesh()
            store_metadata(self.path / "metadata_{name}.yaml".format(name=self.name), spec_dict)

        df.LagrangeInterpolator.interpolate(self._data, data)
        # _data = df.interpolate(data, self._boundary_function_space)

        if "hdf5" in self.spec.save_as:
            self._store_field_hdf5(timestep, time, self._data)

        if "xdmf" in self.spec.save_as:
            self._store_field_xdmf(timestep, time, self._data)

        if "checkpoint" in self.spec.save_as:
            self._checkpoint(timestep, time, self._data)
