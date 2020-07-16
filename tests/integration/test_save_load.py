"""Test that we can load the saved data, and get everything back."""
import tempfile

import numpy as np
import dolfin as df

from pathlib import Path

from setup_solver import SubdomainSolver

from post import (
    Saver,
    Loader,
)

from postfields import (
    Field,
)

from postspec import (
    FieldSpec,
    LoaderSpec,
    SaverSpec,
)


def test_save_load():
    """Solve a problem, save the data, load it back and compare."""
    df.set_log_level(100)       # supress dolfin logger

    # Setup solver
    solver = SubdomainSolver(N=32)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "foo"
        casedir = Path(tmpdirname) / "test_pp_casedir"

        # Setup saver
        field_spec = FieldSpec(stride_timestep=10)
        saver_spec = SaverSpec(casedir=str(casedir))
        saver = Saver(saver_spec)

        assert casedir.exists(), "Aha!"

        saver.store_mesh(
            solver.mesh,
            cell_domains=solver.cell_function,
            facet_domains=solver.facet_function
        )
        saver.add_field(Field("u", field_spec))

        # Solver loop
        time_func_dict = {}
        for i, (t, u) in enumerate(solver.solve(0, 100, 1.0)):
            saver.update(t, i, {"u": u})
            time_func_dict[t] = u.copy(True)
        saver.close()

        # Define loader
        loader_spec = LoaderSpec(casedir=str(casedir))
        loader = Loader(loader_spec)
        loaded_mesh = loader.load_mesh()
        loaded_cell_function = loader.load_mesh_function("cell_function")
        loaded_facet_function = loader.load_mesh_function("facet_function")
        print(set(loaded_cell_function.array()))
        print(set(loaded_facet_function.array()))
        assert False

        # Compare mesh and meshfunctions
        assert np.sum(solver.mesh.coordinates() - loaded_mesh.coordinates()) == 0
        assert np.sum(solver.mesh.cells() - loaded_mesh.cells()) == 0
        assert np.sum(solver.cell_function.array() - loaded_cell_function.array()) == 0
        assert np.sum(solver.facet_function.array() - loaded_facet_function.array()) == 0

        # Compare functions and time
        for loaded_u, loaded_t in loader.load_field("u", return_time=True):
            diff = np.sum(time_func_dict[loaded_t].vector().array() - loaded_u.vector().array())
            assert diff == 0, diff


if __name__ == "__main__":
    test_save_load()
