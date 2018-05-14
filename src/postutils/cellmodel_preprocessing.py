"""Specifications from plotting specific cell models."""

import logging

import numpy as np

from collections import (
    namedtuple,
)

from typing import (
    Dict,
    Tuple,
    List,
    Any,
    Generator,
    Mapping,
)

from postspec import PlotSpec


LOGGER = logging.getLogger(name=__name__)


def preprocess_wei(
        data_map: Mapping[str, np.ndarray],
        names: List[str],
        params: Dict[str, float],
        kwargs: Dict[str, Any]
) -> Generator[Tuple[np.ndarray, PlotSpec], None, None]:
    """
    Scale the inputs so they are appropriate to plot.

    Args:
        data_map: A map from a nme to an array.
        names: The keys in `data_map`.
        params: The parameters `vol` and `beta0` from the solver.
        kwargs: `PotSpec` keyword arguments.

    Return an iterator of the scaled data and corresponding plot spec.
    """
    msg = "Need the volume to compute skalings for all the ions,"""
    assert "Voli" in names, msg

    # TODO: Move this to some global place?
    # NB! This relies on built-in dicts being ordered
    variable_dict = {
        "V": (r"Transmembrane potential", "mV"),
        "m": (r"Voltage Gate (m)", "mV"),
        "h": (r"Voltage Gate (n)", "mV"),
        "n": (r"Voltage Gate (h)", "mV"),
        "NKo": (r"Extracellular Potassium $[K^+]$", "mol"),
        "NKi": (r"Intracellular Potessium $[K^+]$", "mol"),
        "NNao": (r"Extracellular Sodium $[Na^+]$", "mol"),
        "NNai": (r"Intracellular Sodium $[Na^+]$", "mol"),
        "NClo": (r"Exctracellular Chlorine $[Cl^-]$", "mol"),
        "NCli": (r"Intracellular Chlorine $[CL^-]$", "mol"),
        "Voli": (
            r"Ratio of intracellular to extracellular volume",
            r"$Vol_i/Vol_e$"
        ),
        "O": (r"Extracellular Oxygen $[O_2]$", "mol")
    }

    vol = params["vol"]         # Initial colume
    beta0 = params["beta0"]     # Initial volume ratio
    voli = data_map["Voli"]
    volo = (1 + 1/beta0)*vol * voli

    for name in names:
        data = data_map[name]

        if name == "NKo":
            data /= volo
        if name == "NKi":
            data /= voli

        if name == "NNao":
            data /= volo
        if name == "NNai":
            data /= voli

        if name == "NClo":
            data /= volo
        if name == "NClo":
            data /= voli

        plot_spec_kwargs = {
            "name": name,
            "title": name,
            "ylabel": variable_dict[name][1],
            "xlabel": variable_dict[name][0]
        }
        plot_spec_kwargs.update(kwargs)
        yield data, PlotSpec(**plot_spec_kwargs)
