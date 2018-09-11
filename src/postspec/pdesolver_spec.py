from typing import (
    NamedTuple,
    Tuple,
)


#class PDESolverSpec(NamedTuple):
#    theta: float = 0.5
#    ode_scheme: str = "RK4"
#    linear_solver_type: str = "iterative"
#    linear_solver: str = "gmres"
#    preconditioner: str = "petsc_amg"
#    avg_u_constraint: bool = False
#    pde_stimulus: bool = False
#    krylov_absolute_tolerance: float = 1e-15
#    krylov_relative_tolarance: float = 1e-15
#    krylov_nonzero_initial_guess: bool = False


#class SolutionFieldSpec(NamedTuple):
#    save = True
#    save_as = ("hdf5", "xdmf")
#    plot = False
#    start_timestep = -1
#    stride_timestep = 1


#class PDESimulationSpec(NamedTuple):
#    end_time 
#    timestep
#    solution_direcctory = "results"
