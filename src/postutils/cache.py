"""Functions for loading and caching dataframes.

NB! This module is intended to save arrays.
"""

import pandas as pd

from pathlib import Path

from typing import (
    Dict,
    Any,
)


def load_cache(asarray: bool = True, cache_name: str = ".cache") -> pd.DataFrame:
    """Load and return the cache `cache_name`."""
    if asarray:
        return pd.read_pickle(".cache", protocol="bz2").to_records(index=False)
    return pd.read_pickle(".cache", protocol="bz2")


def save_cache(
        data_dict: Dict[str, Any],
        cache_name: str = ".cache",
        clean_cache: bool = False
) -> None:
    """Save the data in a cache. Updates the cache if one with that name exists.

    data_dict is expected to be on the form (`column name`: `column`).
    """
    if Path(cache_name).exists() and not clean_cache:
        df = load_cache(asarray=False, cache_name=cache_name)
    else:
        df = pd.DataFrame()

    for column_name, column in data_dict.items():
        df[column_name] = column

    df.to_pickle(cache_name, protocol="bz2")
