"""Collection of gloabal parameters."""

from typing import Dict

from xalbrain import (
    parameters,
)

import matplotlib as mpl


def set_matplotlib_parameters(kwargs: Dict[str, str] = None) -> None:
    """For nice global matplotlib parameters."""
    font_spec = {
        "family": "ubuntu",
    }
    font_spec.update(kwargs)
    mpl.rc('font', **font_spec)


def set_compilation_parameters() -> None:
    """For computing faster."""
    parameters["form_compiler"]["cpp_optimize"] = True
    flags = "-O3 -ffast-math -march=native"
    parameters["form_compiler"]["cpp_optimize_flags"] = flags
