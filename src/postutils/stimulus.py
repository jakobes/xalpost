from math import sin, cos, pi, asin, acos


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

    # retval = int(sin_curve >= sin(1/(4*frequency) - pulse_width/2))
    # other_pulse = int(sin_curve <= -sin(1/(4*frequency) - pulse_width/2))
    if bidirectional:
        retval -= other_pulse
    else:
        retval += other_pulse
    retval *= amplitude
    return retval


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
