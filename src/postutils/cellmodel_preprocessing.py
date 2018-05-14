"""Specifications from plotting specific cell models."""

import logging

import numpy as np

from collections import (
    namedtuple,
)

from typing import (
    Dict,
    Tuple,
    Any,
    Generator,
)

from postspec import PlotSpec


LOGGER = logging.getLogger(name=__name__)


def preprocess_wei_variable(
        structured_array: np.ndarray,
        params: Dict[str, float],
        kwargs: Dict[str, Any]
) -> Generator[Tuple[np.ndarray, PlotSpec]]:
    """
    Scale the inputs so they are appropriate to plot.

    Args:
        structured_array: An array with the names of the parameters as dtype.
        params: The parameters `vol` and `beta0` from the solver.
        kwargs: `PotSpec` keyword arguments.

    Return an iterator of the scaled data and corresponding plot spec.
    """
    msg = "Need the volume to compute skalings for all the ions,"""
    assert "voli" in structured_array.dtype, msg

    # TODO: Move this to some global place?
    # NB! This relies on built-in dicts being ordered
    variable_dict = {
        "V": (r"Transmembrane potential", "mV"),
        "m": (r"Voltage Gate (m)", "mV"),
        "h": (r"Voltage Gate (n)", "mV"),
        "n": (r"Voltage Gate (h)", "mV"),
        "Ko": (r"Extracellular Potassium $[K^+]$", "mol"),
        "Ki": (r"Intracellular Potessium $[K^+]$", "mol"),
        "Nao": (r"Extracellular Sodium $[Na^+]$", "mol"),
        "Nai": (r"Intracellular Sodium $[Na^+]$", "mol"),
        "Clo": (r"Exctracellular Chlorine $[Cl^-]$", "mol"),
        "Cli": (r"Intracellular Chlorine $[CL^-]$", "mol"),
        "beta": (
            r"Ratio of intracellular to extracellular volume",
            r"$Vol_i/Vol_e$"
        ),
        "O": (r"Extracellular Oxygen $[O_2]$", "mol")
    }

    vol = params["vol"]         # Initial colume
    beta0 = params["beta0"]     # Initial volume ratio
    voli = structured_array["voli"]
    volo = (1 + 1/beta0)*vol * voli

    for name, _ in structured_array.dtype:
        data = structured_array[name]

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
        yield values[:, i], PlotSpec(**plot_spec_kwargs)
