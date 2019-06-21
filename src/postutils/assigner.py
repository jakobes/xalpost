"""A collection of tools to assign inhomogeneous initial conditions."""

import numpy as np
import dolfin as df

from scipy.spatial import cKDTree

from typing import (
    Callable,
    Tuple,
    Iterable,
    Sequence,
    Union,
)


class NearestEdgeTree:

    def __init__(self, points: Union[np.ndarray, Iterable[np.ndarray]] = None) -> None:
        # List of arrays of points representing eg. borders of a domain.
        self._point_array_list: List[np.ndarray] = []
        self._point_set: np.ndarray = None
        self._lookup_tree: cKDTree = None

        if points is not None:
            self.add_points(points)

    def add_points(self, points: Union[np.ndarray, Iterable[np.ndarray]]) -> None:
        for element in tuple(points):
            self._point_array_list.append(element)

    def build_tree(self) -> None:
        if self._point_set is None:
            self._point_set = np.concatenate(self._point_array_list, axis=0)
        else:
            self._point_set = np.concatenate((self._point_set, *self._point_array_list), axis=0)
        self._point_array_list = []
        self._lookup_tree = cKDTree(self._point_set)

    def query(self, points: np.ndarray):

        if self._lookup_tree is None:
            self.build_tree()
        distances, nearest_points_indices  = self._lookup_tree.query(points, 1)
        return self._point_set[nearest_points_indices], distances


def interpolate_ic(
    time: Sequence[float],
    data: np.ndarray,
    receiving_function: df.Function,
    boundaries: Iterable[np.ndarray],
    wavespeed: float = 1.0
) -> None:
    mixed_func_space = receiving_function.function_space()
    mesh = mixed_func_space.mesh()
    V = df.FunctionSpace(mesh, "CG", 1)    # TODO: infer this somehow

    class InitialConditionInterpolator(df.UserExpression):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            self._ic_func = None

            self._nearest_edge_interpolator = NearestEdgeTree(boundaries)

        def set_interpolator(self, interpolation_function):
            self._ic_func = interpolation_function

        def eval(self, value, x):
            _, r = self._nearest_edge_interpolator.query(x)
            value[0] = self._ic_func(r/wavespeed)    # TODO: 1D for now
            # value[0] = r
            # value[0] = self._ic_func(x[0]/wavespeed)    # TODO: 1D for now

    ic_interpolator = InitialConditionInterpolator()

    # Copy functions to be able to assign to them
    subfunction_copy = receiving_function.split(deepcopy=True)
    for i, f in enumerate(subfunction_copy):
        # from IPython import embed; embed()
        # assert False
        ic_interpolator.set_interpolator(lambda x: np.interp(x, time, data[i, :]))
        assigner = df.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(receiving_function.sub(i), df.project(ic_interpolator, V))

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

    class InitialConditionInterpolator(df.UserExpression):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            self._ic_func = None

        def set_interpolator(self, interpolation_function):
            self._ic_func = interpolation_function

        def eval(self, value, x):
            value[0] = self._ic_func(x[0])    # TODO: 1D for now

    ic_interpolator = InitialConditionInterpolator()

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


if __name__ == "__main__":

    times = np.linspace(0, 1, 10)
    vals = np.zeros((times.size, 2))
    vals[:, 0] = np.sin(times)
    vals[:, 1] = np.cos(times)

    interpIC = InterpolatedInitalCondtion(times, vals)

    for i in range(10):
        foo = interpIC(0.1*i)
        print(foo)
