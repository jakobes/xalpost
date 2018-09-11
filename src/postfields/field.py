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
            element = data.function_space().ufl_element()
            spec_dict["element_family"] = str(element.family())  # e.g. Lagrange
            # spec_dict["element_cell"] = element.cell()           # e.g. triangle
            spec_dict["element_degree"] = element.degree()

            store_metadata(self.path/"metadata_{name}.yaml".format(name=self.name), spec_dict)

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
            filename = self.path/"{name}.hdf5".format(name=self.name)
            fieldfile = dolfin.HDF5File(dolfin.mpi_comm_world(), str(filename), "w")
        fieldfile.write(data, "{name}{timestep}".format(name=self.name))
        self._datafile_cache[key] = fieldfile

    def _store_field_xdmf(
            self,
            timestep: int,
            time: float,
            data: dolfin.Function,
            flush_output: bool = True,
            rewrite_mesh: bool = False,
            share_mesh: bool = True
    ) -> None:
        """Save the function as xdmf per timemstep."""
        key = "xdmf"        # Key to access datafile cache
        if key in self._datafile_cache:
            fieldfile = self._datafile_cache[key]
        else:
            filename = self.path/"{name}.xdmf".format(name=self.name)
            fieldfile = dolfin.XDMFFile(dolfin.mpi_comm_world(), str(filename))
            fieldfile.parameters["rewrite_function_mesh"] = rewrite_mesh
            fieldfile.parameters["functions_share_mesh"] = share_mesh
            fieldfile.parameters["flush_output"] = flush_output

        fieldfile.write(data, float(time))
        self._datafile_cache[key] = fieldfile

    def close(self) -> None:
        """Finalise all computations and close file readers/writers."""
        for _, datafile in self._datafile_cache.items():
            datafile.close()
