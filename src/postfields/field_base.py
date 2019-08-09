"""A basecalss for controlling basic file I/O for dolfin functions."""

from pathlib import Path

import logging
import dolfin

from postspec import FieldSpec

from typing import (
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

        Parameters:
            _path: The output directory
        """
        self._name = name
        self._spec = spec
        self._path: Path = ""       # Is this a sensible default?
        self._first_compute: bool = True
        self._datafile_cache: Dict[str, Any] = {}

    def save_this_timestep(self, timestep: int, time: float) -> bool:
        if timestep < self.spec.start_timestep:
            return False
        if int(timestep) % int(self.spec.stride_timestep) == 0:
            return True
        return False

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

    def close(self) -> None:
        """This function is called when closing `Saver`."""
        pass
