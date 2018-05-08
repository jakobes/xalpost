import logging
import dolfin 

from pathlib import Path

from postspec import (
    FieldSpec,
)

from postutils import store_metadata

from typing import (
    List,
    Dict,
    Any,
)

from .field_base import FieldBaseClass


class Field(FieldBaseClass):
    """
    This class should allow specification of time plots over a point, averages, and other 
    statistics.

    ## Idea
    Create a  plot over time class that allows for an (several) operation(s) on the
    function before saving.

    ## Idea:
    Allow for snapshots of a function

    ## Idea:
    Crrate an interface for making configurable nice plots

    ## Idea
    See what cbcpost did.
    """

    def update(self, timestep: int, time: float, data: dolfin.Function) -> None:
        """Update the data."""
        if timestep < self.spec.start_timestep:
            return
        if int(timestep) % int(self.spec.stride_timestep) != 0:
            return

        if self.first_compute:
            self._path.mkdir(parents=False, exist_ok=True)
            spec_dict = self.spec._asdict() 
            element = str(data.function_space().ufl_element())
            spec_dict["element"] = element
            store_metadata(self.path/f"metadata_{self.name}.yaml", spec_dict)
            self.first_compute = False
        
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
        key ="hdf5"
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
        key = "xdmf"
        if key in self._datafile_cache:
            fieldfile = self._datafile_cache[key]
        else:
            filename = self.path/f"{self.name}.xdmf"
            fieldfile = dolfin.XDMFFile(dolfin.mpi_comm_world(), str(filename))
        fieldfile.write(data, float(time))
        self._datafile_cache[key] = fieldfile

    def load_field(self, mesh, timesteps: List[int]):
        # TODO: Does this work?
        # TODO: This needs more thought
        element = dolfin.FiniteElement(
            self.spec.element_family,
            self.spec.element_cell,
            self.sepc.element_degree
        )
        V = FunctionSpace(mesh. element)
        v = function(V)
        
        filename = self.path/f"{self.name}.hdf5"
        with dolfin.HDF5File(mesh.mpi_comm(), filename, "r") as file_handle:
            for ts in timesteps:
                if ts >= self.spec.start_timestep and ts % stride_timestep == 0:
                    file_handle.read(v, f"/{name}{i}")
                    yield v

