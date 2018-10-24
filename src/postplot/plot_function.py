"""A collection of tools for plotting FEniCS entities.

Some of these functions are borroewd from
https://bitbucket.org/fenics-project/dolfin/issues/455/add-ipython-compatible-matplotlib-plotting
"""

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def mesh2triang(mesh: df.Mesh) -> tri.Triangulation:
    """Return an unstructured grid with the connectivity given by matplotlib

    Code borrowed from Chris Richardson
    """
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def mplot_cellfunction(cell_function: df.MeshFunction) -> plt.Figure:
    """Return a pseudocolor plot of an unstructured triangular grid."""
    fig, ax = plt.subplots(1)
    tri = mesh2triang(cell_function.mesh())
    ax.tripcolor(tri, facecolors=cell_function.array())
    return fig


def plot_mesh_triangulation(meshtriang: df.Mesh) -> plt.Figure:
    """Plot the meh as an unstructured grid."""
    fig, ax = plt.subplots(1)
    ax.triplot(triang, 'ko-', lw=1)
    return fig


def mplot_function(function: df.Function) -> plt.Figure:
    mesh = function.function_space().mesh()
    if mesh.geometry().dim() != 2:
        raise AttributeError("Mesh must be 2D")

    fig, ax = plt.subplots(1)

    # DG0 cellwise function
    if function.vector().size() == mesh.num_cells():
        colour_array = function.vector().array()
        ax.tripcolor(mesh2triang(mesh), colour_array)

    # Scalar function, interpolated to vertices
    elif function.value_rank() == 0:
        colour_array = function.compute_vertex_values(mesh)
        ax.tripcolor(mesh2triang(mesh), colour_array, shading="gouraud")

    # Vector function, interpolated to vertices
    elif function.value_rank() == 1:
        vertex_values = function.compute_vertex_values(mesh)
        if len(vertex_values != 2*mesh.num_vertices()):
            raise AttributeError("Vector field must be 2D")

        X = mesh.coordinates()[:, 0]
        Y = mesh.coordinates()[:, 1]
        U = vertex_values[:mesh.num_vertices()]
        V = vertex_values[mesh.num_vertices():]
        ax.quiver(X, Y, U, V)
    return fig


if __name__ == "__main__":
    # func = get_func()
    # foo = mplot_function(func)
    mesh = df.UnitSquareMesh(10, 10)
    triang = mesh2triang(mesh)
    plt.triplot(triang, 'bo-', lw=1)
    plt.savefig("bar.png")
    # foo.savefig("bar.png")
