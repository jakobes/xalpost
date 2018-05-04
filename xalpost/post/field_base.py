"""A basecalss for controlling basic file I/O for dolfin functions."""

import logging
import dolfin 

from pathlib import Path

from xalpost.spec import (
    FieldSpec,
)

from . import store_metadata

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

    @property.setter
    def path(self, path: Path) -> None:
        """Set relative path."""
        self._path = Path/Path(self._name)

    def update(self, timestep: int, data: dolfin.Function) -> None:
        """Update the data."""
        # TODO: Save the time itself
        if timemstep < self.spec.start_timestep:
            return
        if int(timestep) % int(self.spec.stride_timestep) != 0:
            return

        if self.first_compute:
            spec_dict = self.spec._asdict() 
            element = str(data.function_space().ufl_element())
            spec_dict["element"] = element
            store_metadata(self.path/f"{self.name_metadata}.yaml")
            self.first_compute = False
        
        if "hdf5" in self.spec.save_as:
            self.store_field_hdf5(timestep, data)
        if "xdmf" in self.spec.save_as:
            self.store_field_xdmf(timeste, data)

    def store_field_hdf5(
            self,
            function: dolfin.Function,
            timestep: int
    ) -> None:
        """Save as hdf5."""
        filename = self.path/f"{self.name}.hdf5"
        with dolfin.HDF5File(dolfin.mpi_comm_world(), filename, "w") as fieldfile:
            fieldfile.write(function, f"{name}{timestep}")

    def store_field_xdmf(
            self,
            function: dolfin.Function,
            timestep: int
    ) -> None:
        """Save the function as xdmf per timemstep."""
        filename = self.path/f"{self.name}.hdf5"
        with dolfin.XDMFFile(dolfin.mpi_comm_world(), filename) as fieldfile:
            fieldfile.write(function, f"{self.name}{timestep}")
            fieldfile.parameters["flush_output"] = True
            fieldfile.write(data, float(time))

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
