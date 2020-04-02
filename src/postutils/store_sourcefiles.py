import shutil

from pathlib import Path

from typing import (
    Iterable,
)

import dolfin as df


def store_sourcefiles(local_file_names: Iterable[Path], target_directory: Path) -> None:
    target_directory.resolve()      # Make sure is absolute

    if df.MPI.rank(df.MPI.comm_world) == 0:
        target_directory.mkdir(parents=True, exist_ok=True)

        for local_file in local_file_names:
            target = target_directory / local_file
            shutil.copy(local_file.name, target)
    df.MPI.barrier(df.MPI.comm_world)
