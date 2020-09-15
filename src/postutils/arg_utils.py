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

    data = {key: str(value) for key, value in args.__dict__.items()}

    with (out_path / "args.json").open("w") as out_file:
        json.dump(data, out_file, indent=2)
