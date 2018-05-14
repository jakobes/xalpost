"""Utilities for converting arrays to given dtype."""

import numpy as np

from typing import (
    List,
    Tuple,
)


def as_mytype(
        data: np.ndarray,
        new_type: List[Tuple[str, str]],
        mask: np.ndarray = None
) -> np.ndarray:
    """Convert array to a structured array withspecified type.

    NB! `new_type` and `mask` must be compatible.

    Args:
        data: Multidimensional array.
        new_type: A valid dtype for a numpy structured array,
        mask: A mask of the innermost dimension of `data` in case not all rows will be 
            used. default is `np.ones(data.shape[1], dtype=bool)`.
    """
    if mask is None:
        mask = np.ones(data.shape[1], dtype=bool)

    new_array = np.empty(shape=data.shape[0], dtype=new_type)
    for i, (key, _) in enumerate(new_type):
        new_array[key] = new_array[..., mask][:, i]
    return new_array
