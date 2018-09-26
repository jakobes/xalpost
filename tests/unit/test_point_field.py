import pytest
import tempfile

import numpy as np
import dolfin as df

from postfields import PointField
from postspec import FieldSpec
from pathlib import Path


def test_point_field(function):
    points = np.array([0.5, 0.5])
    name = "_test"
    pf = PointField(name, FieldSpec(), points)

    with tempfile.TemporaryDirectory() as tmpdirname:
        pf.path = tmpdirname
        function.vector()[:] = 1
        pf.update(0, 0, function)
        function.vector()[:] = 2
        pf.update(1, 1, function)
        pf.close()

        data = np.load(Path(tmpdirname) / name / "probes_{}.npy".format(name))
    assert np.allclose(data, [1, 2])


@pytest.fixture
def function():
    mesh = df.UnitSquareMesh(2, 2)
    function_space = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(function_space)
    return function

if __name__ == "__main__":
    test_point_field(function())
