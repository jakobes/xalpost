import numpy as np
import yaml

from pathlib import Path

from typing import (
    Any,
    Union,
)


def read_point_metadata(path, name) -> Any:
    try:
        with open(path / Path("{}/metadata_{}.yaml".format(name, name)), "r") as if_handle:
            return yaml.load(if_handle)
    except FileNotFoundError as e:       # TODO: Is this the correct error?
        print(e)
        print("Could not find metadata")


def read_point_values(path, name) -> np.ndarray:
    try:
        with open(path / Path("{}/probes_{}.txt".format(name, name)), "r") as if_handle:
            data = np.array([
                np.fromiter(line.strip().split(","), dtype="f8") for line in if_handle.readlines()
            ])
        return data
    except FileNotFoundError as e:
        print(e)


def load_times(path: Union[str, Path]) -> np.ndarray:
    """Read the times and return them as a numpy array."""
    try:
        with open(path, "r") as if_handle:
            return np.fromiter(if_handle.read().split(), dtype="f8")
    except FileNotFoundError as e:
        print(e)
