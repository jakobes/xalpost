"""Simple wrapper around `dolfin.Function`."""

import logging
import dolfin

from postspec import FieldSpec

from postutils import store_metadata

from typing import (
    List,
    Iterable,
)

from .field_base import FieldBaseClass


LOGGER = logging.getLogger(__name__)


class Field(FieldBaseClass):
    """Store a time series of `dolfin.Function` as xdmf and or hdf5."""

    def update(self, timestep: int, time: float, data: dolfin.Function) -> None:
        """Update the data."""
        if timestep < self.spec.start_timestep:
            return
        if int(timestep) % int(self.spec.stride_timestep) != 0:
            return

        if self.first_compute:
            self.first_compute = False
            self._path.mkdir(parents=False, exist_ok=True)

            # Update spec with element specifications
            spec_dict = self.spec._asdict()
            element = str(data.function_space().ufl_element())
            spec_dict["element"] = element
            store_metadata(self.path/f"metadata_{self.name}.yaml", spec_dict)

        if "hdf5" in self.spec.save_as:
            self._store_field_hdf5(timestep, time, data)
        if "xdmf" in self.spec.save_as:
            self._store_field_xdmf(timestep, time, data)

    def _store_field_hdf5(
            self,
            timestep: int,
            time: float,
            data: dolfin.Function
    ) -> None:
        """Save as hdf5."""
        key = "hdf5"
        if key in self._datafile_cache:
            fieldfile = self._datafile_cache[key]
        else:
            filename = self.path/f"{self.name}.hdf5"
            fieldfile = dolfin.HDF5File(dolfin.mpi_comm_world(), str(filename), "w")
        fieldfile.write(data, f"{self.name}{timestep}")
        self._datafile_cache[key] = fieldfile

    def _store_field_xdmf(
            self,
            timestep: int,
            time: float,
            data: dolfin.Function
    ) -> None:
        """Save the function as xdmf per timemstep."""
        key = "xdmf"        # Key to access datafile cache
        if key in self._datafile_cache:
            fieldfile = self._datafile_cache[key]
        else:
            filename = self.path/f"{self.name}.xdmf"
            fieldfile = dolfin.XDMFFile(dolfin.mpi_comm_world(), str(filename))
            fieldfile.parameters["flush_output"] = True

        fieldfile.write(data, float(time))
        self._datafile_cache[key] = fieldfile

    def load_field(self, mesh, timesteps: List[int]) -> Iterable[dolfin.Function]:
        """Load a field specified by spec."""
        # TODO: Does this work?
        # TODO: This needs more thought

        # Reconstruct Finite Element. TODO: Does not support VectorElement?
        element = dolfin.FiniteElement(
            self.spec.element_family,
            self.spec.element_cell,
            self.sepc.element_degree
        )
        V = dolfin.FunctionSpace(mesh, element)
        v = dolfin.Function(V)

        filename = self.path/f"{self.name}.hdf5"
        with dolfin.HDF5File(mesh.mpi_comm(), filename, "r") as file_handle:
            for ts in timesteps:
                start_test = ts >= self.spec.start_timestep
                stride_test = ts % self.spec.stride_timestep == 0
                if start_test and stride_test:
                    file_handle.read(v, f"/{self.name}{i}")
                    yield v

    def finalise(self) -> None:
        """Finalise all computations and close file readers/writers."""
        for _, datafile in self._datafile_cache.items():
            datafile.close()
