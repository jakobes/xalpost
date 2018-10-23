"""A baseclass for all file I/O."""

from pathlib import Path

import logging

from typing import (
    Dict,
    Any,
)

from postfields.field_base import FieldBaseClass


LOGGER = logging.getLogger(__name__)


class PostProcessorBaseClass:
    """Baseclass for post processors."""

    def __init__(self, spec: Any) -> None:      # TODO: Make spec typoe saver/loader base class
        """Store parameters."""
        self._spec = spec
        self._casedir = Path(spec.casedir)
        self._fields: Dict[str, FieldBaseClass] = {}

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir
