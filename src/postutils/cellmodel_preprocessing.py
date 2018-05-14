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
)

from postspec import PlotSpec


LOGGER = logging.getLogger(name=__name__)


def preprocess_wei(
        values: np.ndarray,
        params: Dict[str, float],
        kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, PlotSpec]:
    """
    Preprocess the Wei cell model for plotting.
    
    Values are assumed to be in the order used in the paper.

    Args:
        values: Array of dimension [N, 12], where N is the number of time steps
        params: A dict with the field 'vol' and 'beta0' used to solve the model.
        kwargs: Keyword arguments passed to PlotSpec.
    """
    assert values.shape[1] == 12, "Expecting Wei to have 12 variables."
    vol = params["vol"]
    beta0 = params["beta0"]
    voli = values[:, 10]
    volo = (1 + 1/beta0)*vol * voli

    # Rescale the ion concentration by volume to get "mol".
    values[:, 4] /= volo
    values[:, 5] /= voli
    values[:, 6] /= volo
    values[:, 7] /= voli
    values[:, 8] /= volo
    values[:, 9] /= voli
    values[:, 10] /= volo

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

    for i, name in enumerate(variable_dict):
        plot_spec_kwargs = {
            "name": name,
            "title": name,
            "ylabel": variable_dict[name][1],
            "xlabel": variable_dict[name][0]
        }
        plot_spec_kwargs.update(kwargs)
        yield values[:, i], PlotSpec(**plot_spec_kwargs)
