import math
import pytest

from postutils import square_pulse

from functools import partial


@pytest.mark.parametrize("bidirectional", [True, False])
def test_square_pulse(bidirectional):
    """Test the suqare pulse."""
    frequency = 2
    pulse_width = 1/16
    amplitude = 2

    test_sq_func = partial(
        square_pulse,
        pulse_width=pulse_width,
        frequency=frequency,
        amplitude=amplitude,
        bidirectional=bidirectional
    )

    positive_times = [
        1/8 - pulse_width/2 + 1e-10,
        1/8 - pulse_width/2 - 1e-10,
        1/8 + pulse_width/2 - 1e-10,
        1/8 + pulse_width/2 + 1e-10,
    ]

    negative_times = [
        3/8 - pulse_width/2 + 1e-10,
        3/8 - pulse_width/2 - 1e-10,
        3/8 + pulse_width/2 - 1e-10,
        3/8 + pulse_width/2 + 1e-10,
    ]

    expected = [2, 0, 2, 0]

    for t, e in zip(positive_times, expected):
        assert test_sq_func(t) == e

    if bidirectional:
        expected = map(lambda x: -1*x, expected)

    for t, e in zip(negative_times, expected):
        assert test_sq_func(t) == e


if __name__ == "__main__":
    test_square_pulse()
