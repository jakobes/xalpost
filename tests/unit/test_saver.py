"""Test the basic functionality of `post.Saver`."""

import dolfin as do

from setup_solver import TestSolver

from post import Saver

from postfields import (
    Field,
)

from postspec import (
    FieldSpec,
    PostProcessorSpec,
    SaverSpec,
)

do.set_log_level(100)
solver = TestSolver(N=32)

field_spec = FieldSpec()
pp_spec = SaverSpec(casedir="test_pp_casedir")

saver = Saver(pp_spec)
saver.store_mesh(solver.mesh)
saver.add_field(Field("u", field_spec))

datafile = do.XDMFFile(do.mpi_comm_world(), "foo.xdmf")
datafile.parameters["rewrite_function_mesh"] = False
datafile.parameters["functions_share_mesh"] = True

for i, (t, u) in enumerate(solver.solve(0, 100, 1.0)):
    print(u.vector().norm("l2"))
    saver.update(t, i, {"u": u})
    datafile.write(u, t)
