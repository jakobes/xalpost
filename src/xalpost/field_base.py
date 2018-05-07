"""A basecalss for controlling basic file I/O for dolfin functions."""

import logging
import dolfin 

from pathlib import Path

from postspec import (
    FieldSpec,
)

from .utils import store_metadata

from typing import (
    List,
    Dict,
    Any,
)


LOGGER = logging.getLogger(__name__)

class FieldBaseClass:
    """A wrapper around dolfin Functions used for the `PostProcessor`."""

    def  __init__(self, name: str, spec: FieldSpec) -> None:
        """Store name and spec.

        Args:
            name: Name of the field.
            spec: Specifications for the field.
        """
        self._name = name
        self._spec = spec
        self._path = None
        self._first_compute = True
        self._datafile_cache = {}

    @property
    def name(self) -> str:
        """Field name."""
        return self._name

    @property
    def spec(self) -> FieldSpec:
        """Field spec."""
        return self._spec

    @property
    def first_compute(self) -> bool:
        """Metadata is stored."""
        return self._first_compute

    @first_compute.setter
    def first_compute(self, b) -> None:
        self._first_compute = b

    @property
    def path(self) -> Path: 
        """Return relative path."""
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        """Set relative path."""
        self._path = path/Path(self._name)

    def update(self, timestep: int, time: float, data: dolfin.Function) -> None:
        """Update the data."""
        # TODO: Save the time itself
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
            self.store_field_hdf5(timestep, time, data)
        if "xdmf" in self.spec.save_as:
            self.store_field_xdmf(timestep, time, data)

    def store_field_hdf5(
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

    def store_field_xdmf(
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

    def finalise(self) -> None:
        for _, datafile in self._datafile_cache.items():
            datafile.close()
