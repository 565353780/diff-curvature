import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes

from diff_curvature.Method.Mesh.geometry import get_faces_coordinates_padded
from diff_curvature.Method.Mesh.feature import face_feature_packed_padded
from diff_curvature.Method.Mesh.dual import get_dual_area_weights_packed
from diff_curvature.Method.Mesh.curvature import (
    get_gaussian_curvature_faces_packed,
    get_mean_curvature_faces_packed,
    get_total_curvature_faces_packed,
)


def sample_points_from_meshes_by_curvature(
    mesh: Meshes,
    num_samples: int,
    return_normals=False,
    mode="gaussian",
    tanh_f=1,
    area_weight=0.1,
):
    """
    Sample points on the mesh surface based on the curvature of the faces.
    The probability of sampling a face is proportional to its curvature.
    """
    B = mesh.verts_padded().shape[0]
    area = mesh.faces_areas_packed().view(-1, 1)

    if mode == "gaussian":
        face_curvature = get_gaussian_curvature_faces_packed(
            mesh, return_density=False
        ).view(-1, 1)
    elif mode == "mean":
        face_curvature = get_mean_curvature_faces_packed(mesh).view(-1, 1)
    elif mode == "total":
        face_curvature = get_total_curvature_faces_packed(mesh).view(-1, 1)

    if tanh_f > 0:
        face_curvature = torch.tanh(tanh_f * face_curvature)
    elif tanh_f < 0:
        assert 1 == 0, "tanh_f must be non-negative"

    face_curvature = face_curvature * area
    probs = face_feature_packed_padded(mesh, face_curvature).view(B, -1)
    probs = probs.abs()
    probs = probs / probs.sum(dim=-1, keepdim=True)

    area_padded = face_feature_packed_padded(mesh, area).view(B, -1)
    area_probs = area_padded / area_padded.sum(dim=-1, keepdim=True)
    probs = (1.0 - area_weight) * probs + area_weight * area_probs

    # Sample faces based on area
    face_indices = torch.multinomial(probs, num_samples, replacement=True)

    # Sample barycentric coordinates

    verts_dual_weight = get_dual_area_weights_packed(mesh)
    verts_dual_weight = verts_dual_weight / verts_dual_weight.sum(dim=-1, keepdim=True)
    verts_dual_weight = face_feature_packed_padded(mesh, verts_dual_weight)

    # Gather the corresponding vertices of the sampled faces
    triangle_coords = get_faces_coordinates_padded(mesh)

    B_idx = torch.tensor(range(B), device=mesh.device).repeat_interleave(num_samples)
    B_idx = B_idx.view(B, num_samples)
    triangle_coords = triangle_coords[B_idx, face_indices]  # (B, num_samples, 3, 3)

    r1 = torch.rand(B, num_samples).to(mesh.device)  # (B, num_samples)
    r2 = torch.rand(B, num_samples).to(mesh.device)  # (B, num_samples)
    u = 1 - r1
    v = r1 * (1 - r2)
    w = r1 * r2

    # Interpolate points using barycentric coordinates
    sampled_points = (
        u[:, :, None] * triangle_coords[:, :, 0]
        + v[:, :, None] * triangle_coords[:, :, 1]
        + w[:, :, None] * triangle_coords[:, :, 2]
    )
    if return_normals:
        triangle_normals = mesh.verts_normals_packed()[mesh.faces_packed()]  # (F, 3, 3)
        triangle_normals = face_feature_packed_padded(
            mesh, triangle_normals
        )  # (B, F, 3, 3)
        triangle_normals = triangle_normals[
            B_idx, face_indices
        ]  # (B, num_samples, 3, 3)

        sampled_normals = (
            u[:, :, None] * triangle_normals[:, :, 0]
            + v[:, :, None] * triangle_normals[:, :, 1]
            + w[:, :, None] * triangle_normals[:, :, 2]
        )

        sampled_normals = F.normalize(sampled_normals, p=2, dim=-1)
        return sampled_points, sampled_normals

    return sampled_points
