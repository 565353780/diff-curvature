import torch
import trimesh
import numpy as np
from torch_scatter import scatter
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian

from diff_curvature.Model.solid_angle import SolidAngle
from diff_curvature.Method.Mesh.geometry import (
    get_faces_coordinates_packed,
    get_faces_angle_packed,
)
from diff_curvature.Method.Mesh.dual import (
    get_dual_area_vertex_packed,
    dual_gather_from_face_features_to_vertices_packed,
    dual_interpolation_from_verts_to_faces_packed,
)


def get_gaussian_curvature_vertices_packed(
    Surfaces: Meshes, return_density=False
) -> torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """
    if isinstance(Surfaces, trimesh.Trimesh):
        face2vertex_index = Surfaces.faces.reshape(-1)
        face2vertex_index = torch.from_numpy(face2vertex_index).long()
    elif isinstance(Surfaces, Meshes):
        face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = get_faces_angle_packed(Surfaces)

    if not isinstance(angle_face, torch.Tensor):
        angle_face = torch.from_numpy(angle_face).float()

    dual_area_per_vertex = get_dual_area_vertex_packed(Surfaces)

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce="sum")
    if return_density:
        curvature = 2 * np.pi - angle_sum_per_vertex
    else:
        curvature = (2 * np.pi - angle_sum_per_vertex) / (dual_area_per_vertex + 1e-8)

    return curvature


### Curvature from faces


def get_gaussian_curvature_faces_packed(
    meshes: Meshes, return_density=False
) -> torch.Tensor:
    """
    Compute the gaussian curvature of each faces in a mesh
    Args:
        meshes: Meshes object
        return_density: bool, if True, return the gaussian curvature density
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of faces
        the gaussian curvature of a face is defined as the solid angle of the face divided by the area of the face
    """

    if isinstance(meshes, trimesh.Trimesh):
        face_coord_packed = get_faces_coordinates_packed(meshes)
        face_coord_packed = torch.from_numpy(face_coord_packed).float()
        trg_mesh_face_normal = meshes.vertex_normals
        trg_mesh_face_normal = trg_mesh_face_normal[meshes.faces, :]
        trg_mesh_face_normal = torch.from_numpy(trg_mesh_face_normal).float()
        face_area = torch.from_numpy(meshes.area_faces).float()

    elif isinstance(meshes, Meshes):
        face_coord_packed = get_faces_coordinates_packed(meshes)
        trg_mesh_normal = meshes.verts_normals_packed()
        trg_mesh_face_normal = trg_mesh_normal[meshes.faces_packed(), :]
        face_area = meshes.faces_areas_packed()

    normal_area = SolidAngle()(trg_mesh_face_normal)

    if return_density:
        return normal_area.view(-1, 1)
    else:
        return (normal_area / (face_area + 1e-10)).view(-1, 1)


def get_gaussian_curvature_vertices_from_face_packed(mesh: Meshes, mode="dual"):
    """
    Rather than computing the gaussian curvature at the vertices, we can compute it at the faces and then gather it to the vertices.
    Args:
        mesh: Meshes object
        mode: str, either 'mean' or 'dual'. If 'dual', use dual area weighting.
    """
    gc_face_packed = get_gaussian_curvature_faces_packed(mesh).view(-1, 1)
    gc_vertex_packed = dual_gather_from_face_features_to_vertices_packed(
        mesh, gc_face_packed, mode=mode
    )
    return gc_vertex_packed


def get_mean_curvature_vertices_packed(mesh: Meshes):
    """
    Compute the mean curvature at the vertices of a mesh.
    """
    if isinstance(mesh, trimesh.Trimesh):
        meshes = Meshes(
            verts=[torch.tensor(mesh.vertices).float()],
            faces=[torch.tensor(mesh.faces)],
        )
    elif isinstance(mesh, Meshes):
        meshes = mesh

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    L, inv_areas = cot_laplacian(verts_packed, faces_packed)
    L = L / 2

    L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
    mean_curvature_vector = (
        (L.mm(verts_packed) - L_sum * verts_packed) * 0.5 * inv_areas * 3
    )
    mean_curvature = -(mean_curvature_vector * meshes.verts_normals_packed()).sum(
        dim=-1
    )
    return mean_curvature.view(-1, 1)


def get_mean_curvature_faces_packed(meshes):
    """
    Compute the mean curvature of the mesh faces.
    """
    mean_curvature = get_mean_curvature_vertices_packed(meshes).view(-1, 1)
    mean_curvature_faces = dual_interpolation_from_verts_to_faces_packed(
        meshes, mean_curvature
    ).squeeze(-1)

    return mean_curvature_faces


def get_total_curvature_vertices_packed(meshes):
    """
    Compute the total curvature of the mesh vertices.
    """
    gaussian_curvature = get_gaussian_curvature_vertices_from_face_packed(meshes).view(
        -1, 1
    )
    mean_curvature = get_mean_curvature_vertices_packed(meshes).view(-1, 1)
    total_curvature = 4 * mean_curvature**2 - gaussian_curvature * 2

    total_curvature = total_curvature.view(-1, 1)
    total_curvature = torch.clamp(total_curvature, min=0)

    return total_curvature / 2


def get_total_curvature_faces_packed(meshes):
    """
    Compute the total curvature of the mesh faces.
    """
    gaussian_curvature = get_gaussian_curvature_faces_packed(
        meshes, return_density=False
    ).view(-1, 1)
    mean_curvature = get_mean_curvature_faces_packed(meshes).view(-1, 1)
    total_curvature_faces = 4 * mean_curvature**2 - gaussian_curvature * 2
    total_curvature_faces = torch.clamp(total_curvature_faces, min=0)
    return total_curvature_faces / 2
