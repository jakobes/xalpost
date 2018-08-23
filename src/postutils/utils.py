"""Utilities for storing metadata."""

from pathlib import Path

from typing import (
    Any,
    Dict,
)

import yaml


def store_metadata(
        filepath: Path,
        meta_dict: Dict[Any, Any],
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


def load_metadata(filepath: Path) -> Dict[str, Any]:
    """Read the metadata associated with a field name.

    Arguments:
        filepath: name of metadata  yaml file.
    """
    with open(filepath, "r") as in_handle:
        return yaml.load(in_handle)


def import_fenicstools() -> Any:
    """Delayed import of fenicstools."""
    import fenicstools
    return fenicstools
