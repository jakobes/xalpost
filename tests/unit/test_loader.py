"""Test the basic functionality of `post.Loader`."""

import dolfin as do

from post import Loader

from postspec import (
    LoaderSpec,
    FieldSpec,
)


field_spec = FieldSpec()
loader_spec = LoaderSpec(casedir="test_pp_casedir")

loader = Loader(loader_spec)
mesh = loader.load_mesh()

cell_function = loader.load_mesh_function(mesh, "CellDomains")
facet_function = loader.load_mesh_function(mesh, "FacetDomains")
print(set(cell_function.array()))
print(set(facet_function.array()))

print(mesh.num_vertices())

md = loader.load_metadata("u")
print(md)

for t, f in loader.load_field("u"):
    print(t, f.vector().norm("l2"))

print(loader.casedir)
