"""Set up solvers for use in testing file io."""

import dolfin as df

from typing import Generator


class PoissonSolver:
    """A solver for the poisson equation for io testing purposes."""

    def __init__(self, N: int) -> None:
        """Initialise the Poisson equation."""
        self._mesh = df.UnitSquareMesh(N, N)
        self.V_space = df.FunctionSpace(self._mesh, "Lagrange", 1)
        u = df.TrialFunction(self.V_space)
        v = df.TestFunction(self.V_space)

        def _boundary(x):
            DOLFIN_EPS = df.DOLFIN_EPS
            return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

        u0 = df.Constant(0.0)
        self.bc = df.DirichletBC(self.V_space, u0, _boundary)

        self.time = df.Constant(0)
        f = df.Expression(
            "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)*10*t",
            t=self.time,
            degree=2
        )

        g = df.Expression("sin(5*x[0])", degree=2)
        self.a = df.inner(df.grad(u), df.grad(v))*df.dx
        self.L = f*v*df.dx + g*v*df.ds

    def _step(self) -> None:
        """solve one time step."""
        df.solve(self.a == self.L, self.u, self.bc)

    @property
    def mesh(self) -> df.Mesh:
        return self._mesh

    def solve(
            self,
            t0: float,
            t1: float,
            timestep: float
    ) -> Generator[df.Function, None, None]:
        """yield solutions.

        Arguments:
            t0: Start time
            t1: Stop time
            timestep: timestep
        """
        t = t0
        self.u = df.Function(self.V_space)
        while t < t1 + 1e-5:
            t += timestep
            self.time.assign(t)
            self._step()
            yield t, self.u


class SubdomainSolver:
    """Solve a Poisson equation with subdomain."""

    def __init__(self, N: int = 32):
        """Initialise solver."""
        self._mesh = df.UnitSquareMesh(N, N)
        self.V_space = df.FunctionSpace(self._mesh, "Lagrange", 1)
        u = df.TrialFunction(self.V_space)
        v = df.TestFunction(self.V_space)


        class Left(df.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5


        class LeftBoundary(df.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 and on_boundary


        left = Left()
        left_boundary = LeftBoundary()

        self.domains = df.MeshFunction("size_t", self._mesh, self._mesh.geometry().dim())
        self.domains.set_all(0)
        left.mark(self.domains, 11)

        self.facets = df.MeshFunction("size_t", self._mesh, self._mesh.geometry().dim() - 1)
        self.facets.set_all(0)
        left.mark(self.facets, 1)

        u0 = df.Constant(0.0)
        self.bc = df.DirichletBC(self.V_space, u0, self.facets, 1)

        dx = df.Measure('dx', domain=self._mesh, subdomain_data=self.domains)
        ds = df.Measure('ds', domain=self._mesh, subdomain_data=self.facets)

        self.time = df.Constant(0)
        f = df.Expression(
            "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)*10*t",
            t=self.time,
            degree=2
        )

        g = df.Expression("sin(5*x[0])*t", degree=2, t=self.time)
        self.a = df.inner(df.grad(u), df.grad(v))*df.dx()
        self.L = f*v*df.dx(11) + g*v*ds(0)

    @property
    def cell_function(self):
        """Return cell function."""
        return self.domains

    @property
    def facet_function(self):
        """Return facet function."""
        return self.facets

    @property
    def mesh(self):
        """Return mesh."""
        return self._mesh

    def _step(self):
        df.solve(self.a == self.L, self.u, self.bc)

    def solve(self, t0, t1, dt):
        """Solver the equation in the interval (`t0`, `t1`) with timestep `dt`."""
        t = t0
        self.u = df.Function(self.V_space)
        while t < t1 + 1e-5:
            t += dt 
            self.time.assign(t)
            self._step()
            yield t, self.u


if __name__ == "__main__":
    df.set_log_level(100)
    solver = SubdomainSolver(N=32)
    for i, (t, sol) in enumerate(solver.solve(0, 5, 0.1)):
        print(i, sol.vector().norm("l2"))
