import torch
import trimesh
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from ops.mesh_geometry import (
    vert_feature_packed_padded,
    get_dual_area_vertex_packed,
    get_gaussian_curvature_vertices_from_face_packed,
)

trg_obj = "/home/chli/chLi/Dataset/Objaverse_82K/mesh/000-000/0000ecca9a234cae994be239f6fec552.obj"

verts, faces = load_obj(trg_obj)[:2]
mesh_np = trimesh.load(trg_obj)

trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

# Dual areas of vertices
dual_areas_padded = vert_feature_packed_padded(
    trg_mesh, get_dual_area_vertex_packed(trg_mesh).view(-1, 1)
)

# gaussian curvature of the vertices and topological characteristics
curvature_vertex_packed = get_gaussian_curvature_vertices_from_face_packed(
    trg_mesh
).view(-1, 1)

curvature_vertex_padded = vert_feature_packed_padded(
    trg_mesh, curvature_vertex_packed.view(-1, 1)
)

diff_eulers = curvature_vertex_padded * dual_areas_padded

diff_euler = diff_eulers.sum(dim=1).view(-1) / 2 / np.pi

print(torch.min(diff_eulers))
print(torch.max(diff_eulers))
print(mesh_np.euler_number)

diff_genus = 1 - diff_euler / 2
# discrete Gauss-Bonnet theorem

print(
    "Gauss-Bonnet theorem: integral of gaussian_curvature - 2*pi*X = ",
    diff_euler.cpu().numpy() - 2 * np.pi * mesh_np.euler_number,
)
