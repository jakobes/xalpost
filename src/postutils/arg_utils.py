import typing as tp

import json
import argparse

from pathlib import Path


def store_arguments(*, args: tp.Any, out_path: tp.Optional[Path] = None):
    """Store contents of args to a json file.

    `args` is a argparse namespace with arguments.
    """
    if out_path is None:
        out_path = Path(".")

    with out_path.open("w") as out_file:
        json.dump(args.__dict__, out_file, indent=2)
