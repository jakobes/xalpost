u"""Point eval field.

Point eval relies on fenicstools.Probe

Thanks to Øyvind Evju and cbcpost (bitbucket.org/simula_cbc/cbcpost).
"""

from pathlib import Path


import logging
import dolfin

import numpy as np

from typing import List

from postspec import FieldSpec

from postutils import (
    store_metadata,
    import_fenicstools,
)

from .field_base import FieldBaseClass


LOGGER = logging.getLogger(__name__)


class PointField(FieldBaseClass):
    """Evaluate a function at a predefined set of descrete points."""

    def __init__(self, name: str, spec: FieldSpec, points: np.ndarray) -> None:
        """Store points, name and spec.

        Arguments:
            name: Name of field. See `FieldBaseClass` for more info.
            spec: Specifications related to field I/O. See `postspec.FieldSpec`.
            points: Array of points at which to evaluate the function. The points must be
                of the same dimension as the function.
        """
        super().__init__(name, spec)
        self._points = points
        self._ft = import_fenicstools()     # Delayed import of fenicstools
        self._probes = None                 # Defined in `compute`
        self._results: List[np.ndarray] = []                  # Append probe evaluations

    def before_first_compute(self, data: dolfin.Function) -> None:
        """Create probes."""
        function_space = data.function_space()
        fs_dim = function_space.mesh().geometry().dim()
        point_dim = self._points.shape[-1]
        msg = f"Point of dimension {point_dim} != function space dimension {fs_dim}"
        assert fs_dim == point_dim, msg

        self._probes = self._ft.Probes(self._points.flatten(), function_space)

    def compute(self, data) -> np.ndarray:
        """Return the value of all probes."""
        # FIXME: This probably does npt work in parallel
        # Make sure that `before_first_compute` is called first
        self._probes(data)      # Evaluate all probes.
        results = self._probes.array()
        self._probes.clear()        # Clear or bad things happen!
        return results

        # if dolfin.MPI.rang(dolfin.mpi_comm_world()) != 0:
        #     results = np.array([], dtype=np.float64)

    def update(self, timestep: int, time: float, data: dolfin.Function) -> None:
        """Update the data."""
        if timestep < self.spec.start_timestep:
            return
        if int(timestep) % int(self.spec.stride_timestep) != 0:
            return

        if self.first_compute:              # Setup everything
            self.first_compute = False      # Do not do this again
            self.before_first_compute(data)
            self._path.mkdir(parents=False, exist_ok=True)

            # Update spec with element specifications
            spec_dict = self.spec._asdict()
            element = str(data.function_space().ufl_element())
            spec_dict["element"] = element
            spec_dict["point"] = list(map(tuple, self._points))
            store_metadata(self.path/f"metadata_{self.name}.yaml", spec_dict)

        self._results.append(self.compute(data))

    def finalise(self) -> None:
        """Save the results."""
        np.save(self.path/Path(f"probes_{self.name}"), np.asarray(self._results))
