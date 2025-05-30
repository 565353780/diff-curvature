import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from diff_curvature.Method.Mesh.curvature import (
    get_gaussian_curvature_vertices_packed,
    get_gaussian_curvature_vertices_from_face_packed,
    get_gaussian_curvature_faces_packed,
    get_mean_curvature_vertices_packed,
    get_mean_curvature_faces_packed,
    get_total_curvature_vertices_packed,
    get_total_curvature_faces_packed,
)


def demo():
    trg_obj = "/home/chli/chLi/Dataset/vae-eval/mesh/000.obj"

    verts, faces = load_obj(trg_obj)[:2]

    trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    # gaussian curvature of the vertices and topological characteristics
    curvature_vertex_packed = get_gaussian_curvature_vertices_from_face_packed(trg_mesh)

    print(curvature_vertex_packed.shape)
    print(torch.min(curvature_vertex_packed))
    print(torch.mean(curvature_vertex_packed))
    print(torch.max(curvature_vertex_packed))
    return True
