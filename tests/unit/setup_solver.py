import dolfin as do


from typing import Generator


class TestSolver:
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


if __name__ == "__main__":
    do.set_log_level(100)
    solver = TestSolver(N=32)
    for i, (t, sol) in enumerate(solver.solve(0, 5, 0.1)):
        print(i, sol.vector().norm("l2"))
