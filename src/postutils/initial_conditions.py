import dolfin as df
import typing as tp


def solve_IC(
    mesh: df.Mesh,
    cell_function: df.MeshFunction,
    tag_ic_dict: tp.Dict[int, tp.List[float]],
    dimension: int
) -> df.Function:
    """Solve a Poisson equation in order to assign values from a cell function to a (CG1) function

    NB! This is not precise --- Nah, it is fairly ok with the interpolation
    """
    dX = df.Measure("dx", domain=mesh, subdomain_data=cell_function)
    DG_fs = df.VectorFunctionSpace(mesh, "DG", 0, dim=dimension)

    u = df.TrialFunction(DG_fs)
    v = df.TestFunction(DG_fs)

    sol = df.Function(DG_fs)
    sol.vector().zero()     # Make sure it is initialised to zero

    # NB! For some reason map(int, cell_tags) does not work with the cell function.
    F = 0
    for ct, ic in tag_ic_dict.items():
        F += -df.dot(u, v)*dX(ct) + df.dot(df.Constant(tuple(ic)), v)*dX(ct)

    a = df.lhs(F)
    L = df.rhs(F)

    # TODO: Why keep diagonal and ident_zeros?
    A = df.assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = df.assemble(L)
    solver = df.KrylovSolver("cg", "petsc_amg")
    solver.set_operator(A)
    solver.solve(sol.vector(), b)

    # clamp to int
    # sol.vector()[:] = sol.vector().get_local()

    target_fs = df.VectorFunctionSpace(mesh, "CG", 1, dim=dimension)
    solsol = df.Function(target_fs)
    solsol.interpolate(sol)
    return solsol


if __name__ == "__main__":
    mesh = df.UnitCubeMesh(40, 40, 40)
    cell_function = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
    cell_function.set_all(0)

    df.CompiledSubDomain("x[0] < 0.5").mark(cell_function, 1)
    df.CompiledSubDomain("x[1] > 0.5").mark(cell_function, 2)
    df.CompiledSubDomain("x[2] > 0.25 && x[2] < 0.75").mark(cell_function, 3)

    CSF_IC = tuple([0]*7)

    STABLE_IC = (    # stable
        -6.70340802e+01,
        1.18435132e-02,
        7.03013587e-02,
        9.78136054e-01,
        1.49366709e-07,
        3.95901396e+00,
        1.78009722e+01
    )

    UNSTABLE_IC = (
        -6.06953303e+01,
        2.63773216e-02,
        1.09906468e-01,
        9.49154804e-01,
        7.69181883e-02,
        1.08414264e+01,
        1.89251358e+01
    )

    ic_dict = {
        1: CSF_IC,
        2: STABLE_IC,
        3: UNSTABLE_IC
    }

    sol = solve_IC(mesh, cell_function, ic_dict, dimension=7)
    # sol = indicator_function(mesh, cell_function, [1, 2, 3])
    with df.XDMFFile("sol.xdmf") as sf:
        sf.write(sol)
