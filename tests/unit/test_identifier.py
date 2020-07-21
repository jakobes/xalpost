import pytest
import tempfile

from pathlib import Path

from postutils import simulation_directory


def test_simulation_directory():
    test_parameters1 = {
        "foo": 1,
        "bar": "baz",
        "path": Path("my_path"),
        "dt": 1e-10,
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        simulation_directory_path1 = simulation_directory(
            parameters=test_parameters1,
            home=Path(tmpdirname),
        )

        test_parameters2 = {
            "foo": 1,
            "bar": "baz",
            "path": Path(tmpdirname),
            "dt": 1e-9,
        }

        simulation_directory_path2 = simulation_directory(
            parameters=test_parameters2,
            home=Path(tmpdirname),
        )
        assert simulation_directory_path2 != simulation_directory_path1

        try:
            simulation_directory_path1 = simulation_directory(
                parameters=test_parameters1,
                home=Path(tmpdirname),
            )
        except FileExistsError:
            pass        # Expected fail
        else:
            assert False

        try:
            simulation_directory_path1 = simulation_directory(
                parameters=test_parameters1,
                home=Path(tmpdirname),
                overwrite_data=True,
            )
        except FileExistsError:
            assert False


if __name__ == "__main__":
    test_simulation_directory()

