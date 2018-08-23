"""Test the basic functionality of `post.Saver`."""

import dolfin as do

from setup_solver import (
    TestSolver,
    SubdomainSolver,
)

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
# solver = TestSolver(N=32)
solver = SubdomainSolver(N=64)

field_spec = FieldSpec()
saver_spec = SaverSpec(casedir="test_pp_casedir")

saver = Saver(saver_spec)
saver.store_mesh(
    solver.mesh,
    cell_domains=solver.cell_function,
    facet_domains=solver.facet_function
)
saver.add_field(Field("u", field_spec))

for i, (t, u) in enumerate(solver.solve(0, 100, 1.0)):
    print(i, u.vector().norm("l2"))
    saver.update(t, i, {"u": u})
saver.close()
