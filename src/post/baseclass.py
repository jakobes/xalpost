"""A baseclass for all file I/O."""

from pathlib import Path

import logging

from typing import Dict

from postspec import (
    PostProcessorSpec,
)

from postfields.field_base import FieldBaseClass


LOGGER = logging.getLogger(__name__)


class PostProcessorBaseClass:
    """Baseclass for post processors."""

    def __init__(self, spec: PostProcessorSpec) -> None:
        """Store parameters."""
        self._spec = spec
        self._casedir = Path(spec.casedir)
        #self._fields: Dict[str, FieldBaseClass] = {}
        self._fields = {}

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir
