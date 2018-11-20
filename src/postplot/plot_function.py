"""A collection of tools for plotting FEniCS entities.

Some of these functions are borroewd from
https://bitbucket.org/fenics-project/dolfin/issues/455/add-ipython-compatible-matplotlib-plotting
"""

import dolfin as df
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from typing import (
    Tuple,
    Any
)


def mesh2triang(mesh: df.Mesh) -> tri.Triangulation:
    """Return an unstructured grid with the connectivity given by matplotlib

    Code borrowed from Chris Richardson
    """
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def mplot_cellfunction(cell_function: df.MeshFunction) -> Tuple[plt.Figure, Any]:
    """Return a pseudocolor plot of an unstructured triangular grid."""
    fig, ax = plt.subplots(1)
    tri = mesh2triang(cell_function.mesh())
    ax.tripcolor(tri, facecolors=cell_function.array())
    return fig, ax


def mplot_mesh(meshtriang: df.Mesh) -> Tuple[plt.Figure, Any]:
    """Plot the meh as an unstructured grid."""
    fig, ax = plt.subplots(1)
    ax.triplot(triang, 'ko-', lw=1)
    return fig, ax


def mplot_function(
    function: df.Function,
    vmin=None,
    vmax=None,
    shading="gouraud",
    colourbar=False,
    colourbar_label=None
) -> Tuple[plt.Figure, Any]:
    """Plot a function. The kind of plot depends on the function."""
    mesh = function.function_space().mesh()
    if mesh.geometry().dim() != 2:
        raise AttributeError("Mesh must be 2D")

    fig, ax = plt.subplots(1)

    tpc = None
    # DG0 cellwise function
    if function.vector().size() == mesh.num_cells():
        colour_array = function.vector().array()
        tpc = ax.tripcolor(mesh2triang(mesh), colour_array, vmin=vmin, vmax=vmax)

    # Scalar function, interpolated to vertices
    elif function.value_rank() == 0:
        colour_array = function.compute_vertex_values(mesh)
        tpc = ax.tripcolor(mesh2triang(mesh), colour_array, shading=shading, vmin=vmin, vmax=vmax)

    # Vector function, interpolated to vertices
    elif function.value_rank() == 1:
        vertex_values = function.compute_vertex_values(mesh)
        if len(vertex_values != 2*mesh.num_vertices()):
            raise AttributeError("Vector field must be 2D")

        X = mesh.coordinates()[:, 0]
        Y = mesh.coordinates()[:, 1]
        U = vertex_values[:mesh.num_vertices()]
        V = vertex_values[mesh.num_vertices():]
        tpc = ax.quiver(X, Y, U, V)

    if colourbar is not None and tpc is not None:
        cb = fig.colorbar(tpc)
        if colourbar_label is not None:
            cb.set_label(colourbar_label)

    return fig, ax
