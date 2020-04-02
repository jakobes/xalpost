"""An interface for saving a `Field` as hdf5."""

import dolfin
import logging
import shutil

import numpy as np
import dolfin as df

from postspec import (
    SaverSpec,
    LoaderSpec,
    FieldSpec,
)

from postfields import (
    Field,
)

from pathlib import Path

from typing import (
    Dict,
    Any,
    List,
    Union,
    Iterable
)

from .baseclass import PostProcessorBaseClass


LOGGER = logging.getLogger(__name__)


class Saver(PostProcessorBaseClass):
    """Class for saving stuff."""

    def __init__(self, spec: LoaderSpec) -> None:
        """Store saver specifications."""
        super().__init__(spec)
        self._time_list = []            # Keep track of time points
        self._first_compute = True      # Perform special action after before first save

        # if self._casedir.exists() and self._spec.overwrite_casedir:
        #     shutil.rmtree(self._casedir)

        if df.MPI.rank(df.MPI.comm_world) == 0:
            self._casedir.mkdir(parents=True, exist_ok=self._spec.overwrite_casedir)
        df.MPI.barrier(df.MPI.comm_world)

    def store_mesh(
            self,
            mesh: dolfin.Mesh,
            cell_domains: dolfin.MeshFunction = None,
            facet_domains: dolfin.MeshFunction = None
    ) -> None:
        """Save the mesh, and cellfunction and facet function if provided."""
        filename = self.casedir / Path("mesh.xdmf")
        # with dolfin.XDMFFile(df.MPI.comm_world, str(filename)) as meshfile:

        # if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        # meshfile = dolfin.XDMFFile(mesh.mpi_comm(), str(filename))
        # dolfin.MPI.barrier(dolfin.MPI.comm_world)

        # meshfile.write(mesh)
        # if cell_domains is not None:
        #     meshfile.write(cell_domains)
        #     dolfin.MPI.barrier(dolfin.MPI.comm_world)
        # if facet_domains is not None:
        #     meshfile.write(facet_domains)
        #     dolfin.MPI.barrier(dolfin.MPI.comm_world)
        # dolfin.MPI.barrier(dolfin.MPI.comm_world)

        with dolfin.XDMFFile(mesh.mpi_comm(), str(filename)) as meshfile:
            meshfile.write(mesh)
            if cell_domains is not None:
                meshfile.write(cell_domains)
            if facet_domains is not None:
                meshfile.write(facet_domains)

        # filename = self.casedir/Path("mesh.hdf5")
        # # with dolfin.HDF5File(mesh.mpi_comm(), str(filename), "w") as meshfile:
        # with dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "w") as meshfile:
        #     meshfile.write(mesh, "/Mesh")
        #     if cell_domains is not None:
        #         meshfile.write(cell_domains, "/CellDomains")
        #     if facet_domains is not None:
                # meshfile.write(facet_domains, "/FacetDomains")

    def add_field(self, field: Field) -> None:
        """Add a field to the postprocessor."""
        # FIXME: Figure out the logging system
        msg = "A field with name {name} already exists.".format(name=field.name)
        assert field.name not in self._fields, msg      # TODO: Issue warning, not abort
        field.path = self._casedir
        self._fields[field.name] = field

    def update(
            self,
            time: float,
            timestep: Union[int, dolfin.Constant],
            data_dict: Dict[str, dolfin.Function]
    ) -> None:
        """Store solutions and perform computations for new timestep."""
        self._time_list.append(float(time))    # This time array has to be sent to each field
        for name, data in data_dict.items():
            self._fields[name].update(timestep, time, data)

        filename = self.casedir/Path("times.txt")
        with open(filename, "a") as of_handle:
            of_handle.write("{} {}\n".format(timestep, float(time)))

    def update_this_timestep(self, *, field_names: Iterable[str], timestep: int, time: float) -> bool:
        return any([self._fields[name].save_this_timestep(timestep, time) for name in field_names])

    def store_initial_condition(self, data_dict) -> None:
        time = 0.0
        timestep = 0
        self._time_list.append(time)
        for name, data in data_dict.items():
            self._fields[name].update(timestep, time, data)

        with (self.casedir / Path("times.txt")).open("a") as of_handle:
            of_handle.write("{} {}\n".format(timestep, float(time)))

    def close(self) -> None:
        """Store the times."""
        for _, field in self._fields.items():
            field.close()
