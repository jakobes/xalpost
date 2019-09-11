import hashlib
import datetime

from pathlib import Path

from typing import (
    Dict,
    Any,
)


def simulation_directory(
    *,
    parameters: Dict[Any, Any],
    home: Path,
    directory_name: str = ".simulations",
    key_length: int = 8,
    overwrite_data: bool = False
) -> Path:
    """Create a unique directory in which to store simulations.

    This function will also create a log file mapping the unique hash value to the
    provided parameter list.

    Arguments:
        parameters: Key value pairs of parameters.
        home: Parent folder of the simulation directory.
        directory_name: Name of the simulation directory.
        key_length: Lenght of the identifier key. It is unlikely to cause collisions.
        overwrite_data: Overwrite existing directory. Usefull if restarting simulations.
    """
    # Check that the storage directory exists
    _home = home
    outdirectory = _home / directory_name
    outdirectory.resolve()      # is this necessary? Probably, if home is provided by user
    outdirectory.mkdir(parents=True, exist_ok=True)

    # Create hash of parameters. Truncate to length 8 for readability
    encoder = hashlib.sha1()
    encoder.update(str(parameters).encode())
    hash = encoder.hexdigest()[:key_length]

    # Abort if directory exists
    simulation_directory = outdirectory / hash      # outdirectory is already absolute
    simulation_directory.mkdir(exist_ok=overwrite_data)

    # Create a list mapping hashes to parameters
    with (outdirectory / "simulation_list.txt").open("a") as outfile_handle:
        time_string = "{0: -- %y -- %d - %Y}".format(datetime.datetime.now())
        outfile_handle.write(hash + time_string + "\n")
        for key, value in parameters.items():
            outfile_handle.write(key + " --- " + str(value) + "\n")
        outfile_handle.write("\n"*3)

    return simulation_directory


if __name__ == "__main__":
    params = {"a": 2, "b": 2}
    simulation_directory(params)
