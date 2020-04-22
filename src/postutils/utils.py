"""Utilities for storing metadata."""

from pathlib import Path
import typing as tp
import dolfin as df


import yaml

import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def store_metadata(
        filepath: Path,
        meta_dict: tp.Dict[tp.Any, tp.Any],
        default_flow_style: bool = False
) -> None:
    """Save spec as `filepath`.

    `name` is converted to a `Path` and save relative to `self.casedir`.

    Arguments:
        name: Name of yaml file.
        spec: Anything compatible with pyaml. It is converted to yaml and dumped.
        default_flow_style: use default_flow_style.
    """
    with open(filepath, "w") as out_handle:
        yaml.dump(meta_dict, out_handle, default_flow_style=default_flow_style)


def load_metadata(filepath: Path) -> tp.Dict[str, tp.Any]:
    """Read the metadata associated with a field name.

    Arguments:
        filepath: name of metadata  yaml file.
    """
    with open(filepath, "r") as in_handle:
        return yaml.load(in_handle)


def import_fenicstools() -> tp.Any:
    """Delayed import of fenicstools."""
    import fenicstools
    return fenicstools


def get_mesh(directory: Path, name: str) -> tp.Tuple[df.Mesh, df.MeshFunction, df.MeshFunction]:
    mesh = df.Mesh()
    mesh_name = str(directory / f"{name}.xdmf")
    with df.XDMFFile(mesh_name) as infile:
        infile.read(mesh)

    cell_function_name = directory / f"{name}_cf.xdmf"
    if not cell_function_name.exists():
        cell_function = None
        logging.info(f"Could not read cell function, file '{cell_function_name} does not exist")
    else:
        mvc = df.MeshValueCollection("size_t", mesh, mesh.geometry().dim())
        with df.XDMFFile(str(cell_function_name)) as infile:
            infile.read(mvc)
            # infile.read(mvc, "tetra")
        cell_function = df.MeshFunction("size_t", mesh, mvc)
    return mesh, cell_function


def get_indicator_function(function_path: Path, mesh: df.Mesh, name: str = "indicator") -> df.Function:
    # Has to be the same as in the bidomain solver
    function_space = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(function_space)
    with df.XDMFFile(str(function_path)) as infile:
        infile.read_checkpoint(function, name, 0)

    return function

