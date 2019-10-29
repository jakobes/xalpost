u"""Point eval field.

Point eval relies on fenicstools.Probe

Thanks to Øyvind Evju and cbcpost (bitbucket.org/simula_cbc/cbcpost).
"""

from pathlib import Path
from time import perf_counter


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
        self._points = np.asarray(points)
        if len(self._points.shape) != 2:    # If we have a single point
            self._points.shape = (1, self._points.shape[0])
        self._ft = import_fenicstools()     # Delayed import of fenicstools
        self._probes = None                 # Defined in `compute`
        self._results: List[np.ndarray] = []                  # Append probe evaluations

    def before_first_compute(self, data: dolfin.Function) -> None:
        """Create probes."""
        function_space = data.function_space()
        fs_dim = function_space.mesh().geometry().dim()
        point_dim = self._points.shape[-1]
        msg = "Point of dimension {point_dim} != function space dimension {fs_dim}".format(
            point_dim=point_dim,
            fs_dim=fs_dim,
        )
        assert fs_dim == point_dim, msg

        if self._spec.sub_field_index is not None:
            function_space = data.function_space().sub(self._spec.sub_field_index)
        else:
            function_space = data.function_space()
        self._probes = self._ft.Probes(self._points.flatten(), function_space)

    def compute(self, data) -> np.ndarray:
        """Return the value of all probes."""
        # FIXME: This probably does not work in parallel

        # Make sure that `before_first_compute` is called first
        if self._spec.sub_field_index is None:
            self._probes(data.sub(self._spec.sub_field_index))
        else:
            self._probes(data)
        results = self._probes.array()
        self._probes.clear()        # Clear or bad things happen!
        return results

    def update(self, timestep: int, time: float, data: dolfin.Function) -> None:
        """Update the data."""
        if not self.save_this_timestep(timestep, time):
            return

        if self.first_compute:              # Setup everything
            self.first_compute = False      # Do not do this again
            self.before_first_compute(data)
            self._path.mkdir(parents=False, exist_ok=True)

            # Update spec with element specifications
            spec_dict = self.spec._asdict()
            element = data.function_space().ufl_element()

            spec_dict["element_family"] = str(element.family())  # e.g. Lagrange
            spec_dict["element_degree"] = element.degree()

            plist = [tuple(map(float, p)) for p in self._points]      # TODO: Untested
            spec_dict["point"] = plist

            store_metadata(self.path/"metadata_{name}.yaml".format(name=self.name), spec_dict)

        with open(self.path/Path("probes_{name}.txt".format(name=self.name)), "a") as of_handle:
            _data = self.compute(data)
            if self._points.shape[0] == 1:
                _data = (_data,)
            _data_format_str = ", ".join(("{}",)*(len(_data) + 1))
            of_handle.write(_data_format_str.format(float(time), *_data))
            of_handle.write("\n")

        self._results.append(self.compute(data))
