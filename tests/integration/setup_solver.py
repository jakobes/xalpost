"""Set up solvers for use in testing file io."""

import dolfin as do

from typing import Generator


class PoissonSolver:
    """A solver for the poisson equation for io testing purposes."""

    def __init__(self, N: int) -> None:
        """Initialise the Poisson equation."""
        self._mesh = do.UnitSquareMesh(N, N)
        self.V_space = do.FunctionSpace(self._mesh, "Lagrange", 1)
        u = do.TrialFunction(self.V_space)
        v = do.TestFunction(self.V_space)

        def _boundary(x):
            DOLFIN_EPS = do.DOLFIN_EPS
            return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

        u0 = do.Constant(0.0)
        self.bc = do.DirichletBC(self.V_space, u0, _boundary)

        self.time = do.Constant(0)
        f = do.Expression(
            "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)*10*t",
            t=self.time,
            degree=2
        )

        g = do.Expression("sin(5*x[0])", degree=2)
        self.a = do.inner(do.grad(u), do.grad(v))*do.dx
        self.L = f*v*do.dx + g*v*do.ds

    def _step(self) -> None:
        """solve one time step."""
        do.solve(self.a == self.L, self.u, self.bc)

    @property
    def mesh(self) -> do.Mesh:
        return self._mesh

    def solve(
            self,
            t0: float,
            t1: float,
            timestep: float
    ) -> Generator[do.Function, None, None]:
        """yield solutions.

        Arguments:
            t0: Start time
            t1: Stop time
            timestep: timestep
        """
        t = t0
        self.u = do.Function(self.V_space)
        while t < t1 + 1e-5:
            t += timestep
            self.time.assign(t)
            self._step()
            yield t, self.u


class SubdomainSolver:
    """Solve a Poisson equation with subdomains."""

    def __init__(self, N: int = 32):
        """Initialise solver."""
        self._mesh = do.UnitSquareMesh(N, N)
        self.V_space = do.FunctionSpace(self._mesh, "Lagrange", 1)
        u = do.TrialFunction(self.V_space)
        v = do.TestFunction(self.V_space)

        class Left(do.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5


        class LeftBoundary(do.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 and on_boundary


        left = Left()
        left_boundary = LeftBoundary()

        self.domains = do.CellFunction("size_t", self._mesh)
        self.domains.set_all(0)
        left.mark(self.domains, 11)

        self.facets = do.FacetFunction("size_t", self._mesh)
        self.facets.set_all(0)
        left.mark(self.facets, 1)

        u0 = do.Constant(0.0)
        self.bc = do.DirichletBC(self.V_space, u0, self.facets, 1)

        dx = do.Measure('dx', domain=self._mesh, subdomain_data=self.domains)
        ds = do.Measure('ds', domain=self._mesh, subdomain_data=self.facets)

        self.time = do.Constant(0)
        f = do.Expression(
            "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)*10*t",
            t=self.time,
            degree=2
        )

        g = do.Expression("sin(5*x[0])*t", degree=2, t=self.time)
        self.a = do.inner(do.grad(u), do.grad(v))*do.dx()
        self.L = f*v*do.dx(11) + g*v*ds(0)

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
        do.solve(self.a == self.L, self.u, self.bc)

    def solve(self, t0, t1, dt):
        """Solver the equation in the interval (`t0`, `t1`) with timestep `dt`."""
        t = t0
        self.u = do.Function(self.V_space)
        while t < t1 + 1e-5:
            t += dt 
            self.time.assign(t)
            self._step()
            yield t, self.u


if __name__ == "__main__":
    do.set_log_level(100)
    solver = SubdomainSolver(N=32)
    for i, (t, sol) in enumerate(solver.solve(0, 5, 0.1)):
        print(i, sol.vector().norm("l2"))
