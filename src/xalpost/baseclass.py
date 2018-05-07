"""A baseclass for all file I/O."""

import logging

from postspec import (
    PostProcessorSpec,
)

from pathlib import Path


LOGGER = logging.getLogger(__name__)


class PostProcessorBaseClass:
    """Baseclass for post processors."""

    def __init__(self, spec: PostProcessorSpec) -> None:
        """Store parameters."""
        self._spec = spec
        self._casedir = Path(spec.casedir)

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir
