"""Utilities for storing metadata."""
from pathlib import Path
import typing as tp
import dolfin as df
import numpy as np

import datetime
import yaml

import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def store_metadata(
        filepath: Path,
        meta_dict: tp.Dict[tp.Any, tp.Any],
) -> None:
    """Save spec as `filepath`.

    `name` is converted to a `Path` and save relative to `self.casedir`.

    Arguments:
        name: Name of yaml file.
        spec: Anything compatible with pyaml. It is converted to yaml and dumped.
    """
    with open(filepath, "w") as out_handle:
        yaml.dump(meta_dict, out_handle, default_flow_style=False)


def load_metadata(filepath: Path) -> tp.Dict[str, tp.Any]:
    """Read the metadata associated with a field name.

    Arguments:
        filepath: name of metadata  yaml file.
    """
    with open(filepath, "r") as in_handle:
        return yaml.load(in_handle, Loader=yaml.Loader)


def import_fenicstools() -> tp.Any:
    """Delayed import of fenicstools."""
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    import fenicstools
    try:
        fenicstools.Probe
    except AttributeError as e:
        logging.error("Could not import fenicstools.Probe")
        raise e
    return fenicstools


def get_mesh(
    directory: Path,
    name: str,
    facet_function_name: str = None,
    cell_function_name: str = None
) -> tp.Tuple[df.Mesh, df.MeshFunction, df.MeshFunction]:
    mesh = df.Mesh()
    mesh_name = str(directory / f"{name}.xdmf")
    with df.XDMFFile(mesh_name) as infile:
        infile.read(mesh)

    if cell_function_name is None:
        cell_function_name = f"{name}_cf.xdmf"
    _cell_function_name = directory / cell_function_name
    if not cell_function_name.exists():
        cell_function = None
        logging.info(f"Could not read cell function, file '{_cell_function_name} does not exist")
    else:
        mvc = df.MeshValueCollection("size_t", mesh, mesh.geometry().dim())
        with df.XDMFFile(str(_cell_function_name)) as infile:
            infile.read(mvc)
            # infile.read(mvc, "tetra")
        cell_function = df.MeshFunction("size_t", mesh, mvc)

    if facet_function_name is None:
        facet_function_name = f"{name}_ff.xdmf"
    _facet_function_name = directory / facet_function_name
    if not _facet_function_name.exists():
        facet_function = None
        logging.info(f"Could not read facet function, file '{_facet_function_name} does not exist")
    else:
        mvc = df.MeshValueCollection("size_t", mesh, mesh.geometry().dim() - 1)
        with df.XDMFFile(str(_facet_function_name)) as infile:
            infile.read(mvc)
        facet_function = df.MeshFunction("size_t", mesh, mvc)
    return mesh, cell_function, facet_function


def save_mesh(
    directory: Path,
    name: str,
    *,
    mesh: tp.Optional[df.Mesh] = None,
    cell_function: tp.Optional[df.MeshFunction] = None,
    facet_function: tp.Optional[df.MeshFunction] = None
) -> None:
    if mesh is not None:
        mesh_path = directory / f"{name}.xdmf"
        logger.info(f"Saving mesh as {mesh_path}")
        with df.XDMFFile(str(mesh_path)) as mesh_file:
            mesh_file.write(mesh)
    if cell_function is not None:
        cell_path = directory / f"{name}_cf.xdmf"
        logger.info(f"Saving cell_function as {cell_path}")
        with df.XDMFFile(str(cell_path)) as cell_file:
            cell_file.write(cell_function)
    if facet_function is not None:
        facet_path = directory / f"{name}_ff.xdmf"
        logger.info(f"Saving facet_function as {facet_path}")
        with df.XDMFFile(str(facet_path)) as facet_file:
            facet_file.write(facet_function)


def get_indicator_function(function_path: Path, mesh: df.Mesh, name: str = "indicator") -> df.Function:
    # Has to be the same as in the bidomain solver
    function_space = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(function_space)
    with df.XDMFFile(str(function_path)) as infile:
        infile.read_checkpoint(function, name, 0)

    return function


def get_current_time_mpi() -> datetime.datetime:
    if df.MPI.rank(df.MPI.comm_world) == 0:
        current_time = datetime.datetime.now()
    else:
        current_time = None
    current_time = df.MPI.comm_world.bcast(current_time, root=0)
    return current_time


def save_function(
    indicator_function: df.Function,
    output_path: Path,
    name: tp.Optional[str] = None
) -> None:
    """Save a dolfin function as xdmf checkpoint for reading and visualisation."""
    if name is None:
        name = "indicator"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with df.XDMFFile(str(output_path)) as xdmf:
        xdmf.write_checkpoint(indicator_function, name, 0)


def read_function(
    mesh: df.Mesh,
    name: Path,
    function_name: tp.Optional[str] = None,
    function_space_type: str = "CG"
) -> df.Function:
    if function_name is None:
        function_name = "indicator"
    function_space = df.FunctionSpace(mesh, function_space_type, 1)    # FIXME: I believe it is CG and not DG
    function = df.Function(function_space)

    with df.XDMFFile(str(name)) as xdmf:
        xdmf.read_checkpoint(function, function_name, 0)
    return function


def get_part_number(timestep: int, break_time_step: tp.Optional[int]) -> str:
    if break_time_step is None:
        return ""

    part_number = timestep // break_time_step
    return f"_part{part_number}"


def check_bounds(points: np.ndarray, limit: float = 100) -> bool:
    span = np.max(points, axis=0) - np.min(points, axis=0)
    max_span = np.max(span)
    if max_span > limit:
        raise ValueError(f"Max span is {max_span}. Assuming this is in mm not cm")
    return True
