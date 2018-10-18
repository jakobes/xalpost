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
    name = "_test"
    pf = PointField(name, FieldSpec(), points)

    with tempfile.TemporaryDirectory() as tmpdirname:
        pf.path = tmpdirname
        function.vector()[:] = 1
        pf.update(0, 0, function)
        function.vector()[:] = 2
        pf.update(1, 1, function)
        pf.close()

        data = read_point_values(tmpdirname, "_test")
    assert np.allclose(data, [[1]*len(points), [2]*len(points)])


@pytest.fixture
def function():
    mesh = df.UnitSquareMesh(2, 2)
    function_space = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(function_space)
    return function

if __name__ == "__main__":
    points = np.array([[0.5, 0.5], [0.25, 0.75]])
    points = np.array([0.5, 0.5])
    test_point_field_2(function(), points)

