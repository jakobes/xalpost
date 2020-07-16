import pytest
import tempfile

import numpy as np
import dolfin as df

from postfields import PointField
from postspec import FieldSpec
from pathlib import Path

from post import read_point_values


@pytest.mark.parametrize("points", [
    np.array([[0.5, 0.5], [0.25, 0.75]]),
    np.array([0.5, 0.5])
])
def test_point_field_2(function, points):
    point_field_name = "test"

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)

        pf = PointField(point_field_name, FieldSpec(), points)
        pf.path = tmpdir        # mimick saver behaviour

        # Update the underlying function and the point field twice for time stepping
        function.vector()[:] = 1
        pf.update(timestep=1, time=0.1, data=function)
        function.vector()[:] = 2
        pf.update(timestep=2, time=0.2, data=function)

        msg = (pf.path / f"probes_{point_field_name}.txt")
        assert (pf.path / f"probes_{point_field_name}.txt").exists(), msg

        msg = (tmpdir / f"{point_field_name}" / f"probes_{point_field_name}.txt")
        assert (tmpdir / f"{point_field_name}" / f"probes_{point_field_name}.txt").exists(), msg
        data = read_point_values(
            path=(tmpdir / f"{point_field_name}" / f"probes_{point_field_name}.txt")
        )

    # First element is time, second is data
    assert np.allclose(data, [[0.1] + [1]*len(points), [0.2] + [2]*len(points)])


@pytest.fixture
def function():
    mesh = df.UnitSquareMesh(2, 2)
    function_space = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(function_space)
    return function

if __name__ == "__main__":

    def function():
        mesh = df.UnitSquareMesh(2, 2)
        function_space = df.FunctionSpace(mesh, "CG", 1)
        function = df.Function(function_space)
        return function

    points = np.array([[0.5, 0.5], [0.25, 0.75]])
    test_point_field_2(function(), points)
