import logging
import dolfin 

from pathlib import Path

from postspec import (
    FieldSpec,
)

from .utils import store_metadata

from typing import (
    List,
    Dict,
    Any,
)

from .field_base import FieldBaseClass


class Field(FieldBaseClass):
    """
    This class should allow specification of time plots over a point, averages, and other 
    statistics.

    ## Idea
    Create a  plot over time class that allows for an (several) operation(s) on the
    function before saving.

    ## Idea:
    Allow for snapshots of a function

    ## Idea:
    Crrate an interface for making configurable nice plots

    ## Idea
    See what cbcpost did.
    """
    pass
