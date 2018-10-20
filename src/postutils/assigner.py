"""A collection of tools to assign inhomogeneous initial conditions."""

import numpy as np
import dolfin as df

from typing import (
    Callable,
    Tuple,
    Iterable,
)


class NonuniformIC:
    def __init__(self, coordinates: np.ndarray, data: np.ndarray) -> None:
        """
        TODO: Is this for temporal interpolation?

        Arguments:
            coordinates: Original sampling coordinates for `data`.
            data: Samples values at `coordinates`.
        """
        self.data = data
        cmax = coordinates.max()
        cmin = coordinates.min()
        self.coordinates = (cmax - cmin)*np.linspace(0, 1, data.shape[0]) + cmin

    def __call__(self) -> np.ndarray:
        for i in range(self.data.shape[1]):
            yield lambda x: np.interp(x, self.coordinates, self.data[:, i])


def new_assign_ic(
        receiving_function: df.Function,
        ic_generator: NonuniformIC,
        degree: int = 1
) -> None:
    """
    Assign receiving_function(x, y) <- `ic_function`(x, y), for x, in the mesh.

    Arguments:
        receiving_function: The function which is assigned the initial condition.
        ic_function_tuple: A tuple of python callables which return the initial condition for each
            point (x, y). The number of functions must match the number of subfunctions 
            in `receiving_function`.
    """
    mixed_func_space = receiving_function.function_space()
    mesh = mixed_func_space.mesh()
    V = df.FunctionSpace(mesh, "CG", 1)    # TODO: infer this somehow

    # Copy functions to be able to assign to them
    functions = receiving_function.split(deepcopy=True)

    for i, (f, ic_func) in enumerate(zip(functions, ic_generator())):
        class IC(df.Expression):
            def eval(self, value, x):
                value[0] = ic_func(x[0])    # TODO: 1D for now

        ic = IC(degree=degree)
        assigner = df.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(receiving_function.sub(i), df.project(ic, V))


def assign_ic(func, data):
    mixed_func_space = func.function_space()

    functions = func.split(deepcopy=True)
    V = df.FunctionSpace(mixed_func_space.mesh(), "CG", 1)
    ic_indices = np.random.randint(
        0,
        data.shape[0],
        size=functions[0].vector().local_size()
    )
    _data = data[ic_indices]

    for i, f in enumerate(functions):
        ic_func = df.Function(V)
        ic_func.vector()[:] = np.array(_data[:, i])

        assigner = df.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(func.sub(i), ic_func)


def assign_restart_ic(
        receiving_function: df.Function,
        assigning_func_iterator: Iterable[df.Function]
) -> None:
    """Assign a seriess of functions to the `receiving_function`.

    This function is indended for use when restarting simulations, using previously computed
    solutions as initial conditions.
    """
    # Get receiving function space
    mixed_function_space = receiving_function.function_space( )
    assigning_function_space = df.FunctionSpace(mixed_function_space.mesh(), "CG", 1)

    for subfunc_idx, assigning_sub_function in enumerate(assigning_func_iterator):
        assigner = df.FunctionAssigner(
            mixed_function_space.sub(subfunc_idx),
            assigning_function_space
        )
        assigner.assign(receiving_function.sub(subfunc_idx), assigning_sub_function)
