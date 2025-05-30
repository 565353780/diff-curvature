import trimesh
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from diff_curvature.Method.mesh_geometry import (
    get_gaussian_curvature_vertices_from_face_packed,
)

trg_obj = "/home/chli/chLi/Dataset/Objaverse_82K/mesh/000-000/0000ecca9a234cae994be239f6fec552.obj"

verts, faces = load_obj(trg_obj)[:2]
mesh_np = trimesh.load(trg_obj)

trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

# gaussian curvature of the vertices and topological characteristics
curvature_vertex_packed = get_gaussian_curvature_vertices_from_face_packed(trg_mesh)

print(curvature_vertex_packed.shape)
