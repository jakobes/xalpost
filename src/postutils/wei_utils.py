"""Get initial conditions based on a reference solution."""

import logging

import pickle
import numpy as np
import pandas as pd


from typing import (
    Iterable,
    Tuple,
)

from xalbrain import (
    Function,
)

logger = logging.getLogger(name=__name__)


def get_solution(*_, name: str) -> np.ndarray:
    """Load and return the specified reference solution."""
    filenames = {
        "wei": "REFERENCE_WEI",
    }
    if name in filenames:
        return np.load(f"{filenames[name]}.npy")
    return np.load(name)


def chaotic_ic(data: np.ndarray, N: int, seed: int = 42) -> np.ndarray:
    """Draw N random numbers from `data`."""
    rngesus = np.random.RandomState(seed)
    indices = rngesus.random_integers(0, data.shape[0] - 1, size=N)
    return data[indices]


def wei_uniform_ic(data: np.ndarray, state: str, index: int = None):
    """
    Return a map {function name: initial condition} for the Wei cell model.

    If `index` is specified, `state` is ignored.
    """
    state_dict = {
        "fire": 192340,
        "flat": 201500
    }
    if index is None:
        index = state_dict[state]

    # The names are Wei model specific.
    names = ("V", "m", "h", "n", "NKo", "NKi", "NNao", "NNai", "NClo", "NCli", "vol", "O")
    return {name: val for name, val in zip(names, data[index])}


def create_dataframe(
        solutions: Iterable[Tuple[Tuple[float], Tuple[Function]]],
        names: Tuple[str],
        point: Tuple[float],
        stride: int = 1
) -> pd.DataFrame:
    """
    Evalueate solutiuons at a point and store every `stride` timestep in a dataframe.

    Args:
        solutions: (t0, t1), solution). Here, solutions is a tuple of `Function`. The
            The solutions at each time step.
        point: A point in ND space, where N is the dimentsion of the ssolution function
            space. The functions are evaluated at thhis point and stored in the dataframe.
        stride (optional): Skip every `stride` timestep.
    """
    columns = ["time"] + list(names)
    dataframe = pd.DataFrame(columns=columns)

    for i, (t1, solution) in enumerate(solutions):
        if i % int(stride) == 0:
            # Store the time, f(p0, p1) for each variable in the solution
            dataframe.loc[i] = [t1] + [f(*point) for f in solution]
    return dataframe
