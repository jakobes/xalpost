"""Simple wrapper around `dolfin.Function`."""

import logging
import dolfin

from postspec import FieldSpec

from postutils import (
    store_metadata,
    get_part_number,
)

from pathlib import Path

from typing import (
    List,
    Iterable,
)
from .field_base import FieldBaseClass

import dolfin as df


LOGGER = logging.getLogger(__name__)


# --- I/O stuff ---
class _HDF5Link:
    """Helper class for creating links in HDF5-files."""
    cpp_link_module = None

    def __init__(self):
        cpp_link_code = """
        #include <hdf5.h>

        // dolfin headers
        #include <dolfin/io/HDF5Interface.h>
        #include <dolfin/common/MPI.h>

        // pybind headers
        #include <pybind11/pybind11.h>

        namespace py = pybind11;

        namespace dolfin
        {
        void link_dataset(const MPI_Comm comm,
                          const std::string hdf5_filename,
                          const std::string link_from,
                          const std::string link_to, bool use_mpiio)
        {
            hid_t hdf5_file_id = HDF5Interface::open_file(comm, hdf5_filename, "a", use_mpiio);
            herr_t status = H5Lcreate_hard(hdf5_file_id, link_from.c_str(), H5L_SAME_LOC,
                                link_to.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            dolfin_assert(status != HDF5_FAIL);

            HDF5Interface::close_file(hdf5_file_id);
        }

        PYBIND11_MODULE(SIGNATURE, m) {
            m.def("link_dataset", &link_dataset);
        }

        }   // end namespace dolfin
        """
        # self.cpp_link_module = dolfin.compile_cpp_code(cpp_link_code, additional_system_headers=["dolfin/io/HDF5Interface.h"])
        self.cpp_link_module = dolfin.compile_cpp_code(cpp_link_code)

    def __call__(self, hdf5filename, link_from, link_to):
        "Create link in hdf5file."
        use_mpiio = dolfin.MPI.size(dolfin.MPI.comm_world) > 1
        self.cpp_link_module.link_dataset(0, hdf5filename, link_from, link_to, use_mpiio)

        # TODO: Dolfin uses a custom caster for the MPI communicator. Move to separate extension module
        # self.cpp_link_module.link_dataset(dolfin.MPI.comm_world, hdf5filename, link_from, link_to, use_mpiio)


# hdf5_link = _HDF5Link()


class Field(FieldBaseClass):
    """Store a time series of `dolfin.Function` as xdmf and or hdf5."""

    def update(self, timestep: int, time: float, data: dolfin.Function) -> None:
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

            store_metadata(self.path / "metadata_{name}.yaml".format(name=self.name), spec_dict)

        if "hdf5" in self.spec.save_as:
            self._store_field_hdf5(timestep, time, data)

        if "xdmf" in self.spec.save_as:
            self._store_field_xdmf(timestep, time, data)

        if "checkpoint" in self.spec.save_as:
            self._checkpoint(timestep, time, data)

    def _store_field_hdf5(
            self,
            timestep: int,
            time: float,
            data: dolfin.Function
    ) -> None:
        """Save as hdf5."""
        _key = "hdf5"
        part_annotation = get_part_number(timestep, self._spec.num_steps_in_part)
        filename = self.path / f"{self.name}{part_annotation}.hdf5"
        if filename.exists():
            fieldfile = dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "a")
        else:
            fieldfile = dolfin.HDF5File(dolfin.MPI.comm_world, str(filename), "w")

        # Store the function space information once
        if not fieldfile.has_dataset(self.name):
            fieldfile.write(data, self.name)

        if not fieldfile.has_dataset("mesh"):
            fieldfile.write(data.function_space().mesh(), "mesh")

        fieldfile.write(data.vector(), self.name + str(timestep) + "/vector")

        # Link information about function space from hash-dataset
        # hdf5_link(str(filename), self.name + "/x_cell_dofs", self.name + str(timestep) + "/x_cell_dofs")
        # hdf5_link(str(filename), self.name + "/cell_dofs", self.name + str(timestep) + "/cell_dofs")
        # hdf5_link(str(filename), self.name + "/cells", self.name + str(timestep) + "/cells")
        fieldfile.close()

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
            part_annotation = get_part_number(timestep, self._spec.num_steps_in_part)
            filename = self.path / f"{self.name}{part_annotation}.xdmf"
            fieldfile = dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename))
            fieldfile.parameters["rewrite_function_mesh"] = rewrite_mesh
            fieldfile.parameters["functions_share_mesh"] = share_mesh
            fieldfile.parameters["flush_output"] = flush_output

        fieldfile.write(data, float(time))
        self._datafile_cache[key] = fieldfile

    def _checkpoint(
            self,
            timestep: int,
            time: float,
            data: dolfin.Function,
            flush_output: bool = True,
            rewrite_mesh: bool = False,
            share_mesh: bool = True,
    ) -> None:
        key = "checkpoint"
        if key in self._datafile_cache:
            fieldfile = self._datafile_cache[key]
        else:
            part_annotation = get_part_number(timestep, self._spec.num_steps_in_part)
            filename = self.path / f"{self.name}_chk{part_annotation}.xdmf"
            fieldfile = dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename))
            # fieldfile.parameters["rewrite_function_mesh"] = rewrite_mesh
            # fieldfile.parameters["functions_share_mesh"] = share_mesh
            # fieldfile.parameters["flush_output"] = flush_output

        fieldfile.write_checkpoint(data, self.name, int(timestep), append=True)
        self._datafile_cache[key] = fieldfile

    def load(self):
        return

    def close(self) -> None:
        """Finalise all computations and close file readers/writers."""
        for _, datafile in self._datafile_cache.items():
            datafile.close()
