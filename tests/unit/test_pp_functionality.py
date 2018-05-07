"""Test the basic functionality of thepost processor."""

from dolfin import *

from xalpost import (
    PostProcessor,
    Field,
)

from postspec import (
    FieldSpec,
    PostProcessorSpec,
)

field_spec = FieldSpec()
pp_spec = PostProcessorSpec(casedir="pp_casedir")

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

postprocessor = PostProcessor(pp_spec)
postprocessor.store_mesh(mesh)
postprocessor.add_field(Field("u", field_spec))


class Saver:
    def __init__(self):
        self.xdmf = XDMFFile(dolfin.mpi_comm_world(), "barbaz.xdmf")

    def save(self, x, time):
        self.xdmf.write(x, time)

    def close(self):
        self.xdmf.close()


def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)

time = Constant(0)
f = Expression(
    "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)*t",
    t=time,
    degree=2
)

g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

u = Function(V)

xdmf = XDMFFile(mesh.mpi_comm(), "foobar.xdmf")
saver = Saver()

dt = 1.0
t = 0
for i in range(5):
    t += dt
    time.assign(t)
    solve(a == L, u, bc)
    xdmf.write(u, t)
    saver.save(u, t)
    postprocessor.update(t, i, {"u": u})
xdmf.close()
saver.close()
# postprocessor.finalise()
