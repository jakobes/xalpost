import dolfin as df

from math import sin, pi

from typing import (
    Callable,
    Dict,
    List,
    Any
)


def square_pulse(
        time: float,
        pulse_width: float,
        frequency: float,
        amplitude: float,
        bidirectional: bool=True
) -> float:
    """Unidirectional square pulse.

    NB! Everything is in milli seconds.

    Arguments:
        time: The current time.
        pulse_width: The with of each square pulse.
        frequency: The number of pulses per ms second.
    """
    sin_curve = sin(2*pi*frequency*time)
    threshold = sin(2*pi*frequency*(1/(4*frequency) - pulse_width/2))
    retval = int(sin_curve >= threshold)

    other_pulse = int(sin_curve <= -threshold)
    if bidirectional:
        retval -= other_pulse
    else:
        retval += other_pulse
    retval *= amplitude
    return retval


class Time_expression(df.Expression):
    """Expression wrapper for a time dependent function only accepting *args."""

    def __init__(
            self,
            func: Callable,
            time: df.Constant,
            *args: List[Any],
            area: float=1.0,
            **kwargs: Dict[Any, Any]
    ) -> None:
        """Wrap `func` as a partial accepting only `time`.

        Arguments:
            func: The time dependent function to wrapp.
            time: The internal time of the solver.
            *args: Arguments passed to `func`.
            area: Optionally scale the wrapped function by area. Defaults to 1.
            **kwargs: Arguments passed to `df.Expression`.
        """
        self.area
        self.time = time
        self.func = lambda x: func(x(0), *args)

    def eval(self, value, x) -> None:
        """Evaluate the wrapped func"""
        value[0] = self.func(self.time)/self.area


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    pulse_width = 1.2       # mS
    frequency = 60e-3       # 60 Hz
    amplitude = 3           # mA

    fig = plt.figure()
    ax = fig.add_subplot(111)

    time = np.linspace(0, 3e1, int(1e4))
    sin_curve = amplitude*np.sin(2*pi*frequency*time)
    square = np.vectorize(square_puls)(time, pulse_width, frequency, amplitude)

    ax.plot(time, square)
    ax.plot(time, sin_curve)

    fig.savefig("foo.png")
