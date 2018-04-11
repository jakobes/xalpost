"""A wrapper around dolfin Functions used for the `PostProcessor`."""

import dolfin
import logging

from xalpost.spec import (
    FieldSpec,
)

LOGGER = logging.getLogger(__name__)


class Field:
    """A wrapper around dolfin Functions used for the `PostProcessor`."""

    def  __init__(self, name: str, spec: FieldSpec):
        """Store name and spec.

        Args:
            name: Name of the field.
            spec: Specifications for the field.
        """
        self._name = name
        self._spec = spec
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
