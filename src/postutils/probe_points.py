import numpy as np

from typing import (
    Iterable,
    Tuple
)


def circle_points(
    *,
    radii: Iterable[float],
    num_points: Iterable[int],
    r0: Tuple[float, float] = (0, 0)
) -> np.ndarray:
    x0, y0 = r0
    circles = []

    for r, n in zip(radii, num_points):
        t = np.linspace(0, 2*np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x + x0, y + y0])
    return np.concatenate(circles, axis=0)


def grid_points(*, dimension: int, num_points: int) -> np.ndarray:
    _npj = num_points*1j
    if dimension == 1:
        numbers = np.mgrid[0:1:_npj]
        return np.vstack(map(lambda x: x.ravel(), numbers)).reshape(-1, dimension)
    if dimension == 2:
        # numbers = np.mgrid[0:1:_npj, 0:1:_npj]
        my_range = np.arange(10)/10

        foo = np.zeros(shape=(10, 2))
        foo[:, 0] = my_range

        bar = np.zeros(shape=(10, 2))
        bar[:, 0] = my_range
        bar[:, 1] = my_range
        return np.vstack((foo, bar))
    if dimension == 3:
        assert False, "3D points not supported"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    r = [0, 0.1, 0.2]
    n = [1, 10, 20]
    circles = circle_points(radii=r, num_points=n, r0=(10, 10))

    fig, ax = plt.subplots()
    ax.scatter(circles[:, 0], circles[:, 1])

    ax.set_aspect('equal')
    plt.show()
