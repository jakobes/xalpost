"""A baseclass for all file I/O."""

from xalpost.spec import (
    PostProcessorSpec,
)

from pathlib import Path


class PostProcessorBaseClass:
    def __init__(self, spec: PostProcessorSpec) -> None:
        self._spec = spec
        self._casedir = Path(spec.casedir)

    @property
    def casedir(self) -> Path:
        """Return the casedir."""
        return self._casedir

