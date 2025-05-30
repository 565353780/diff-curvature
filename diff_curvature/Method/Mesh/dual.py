import torch
import trimesh
from torch_scatter import scatter
from einops import rearrange, repeat
from pytorch3d.structures import Meshes

from diff_curvature.Method.Mesh.geometry import get_faces_angle_packed


def get_dual_area_weights_packed(Surfaces: Meshes) -> torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = get_faces_angle_packed(Surfaces)
    if not isinstance(angles, torch.Tensor):
        angles = torch.from_numpy(angles)

    angles_roll = torch.stack(
        [angles, angles.roll(-1, dims=1), angles.roll(1, dims=1)], dim=1
    )

    sinanlge = torch.sin(angles)

    cosdiffangle = torch.cos(angles_roll[..., -2] - angles_roll[..., -1])

    dual_area_weight = sinanlge * cosdiffangle

    dual_area_weight = dual_area_weight / (
        torch.sum(dual_area_weight, dim=-1, keepdim=True) + 1e-8
    )

    dual_area_weight = torch.clamp(dual_area_weight, 0, 1)

    dual_area_weight = dual_area_weight / dual_area_weight.sum(dim=-1, keepdim=True)

    return dual_area_weight


def get_dual_area_vertex_packed(Surfaces: Meshes, return_type="packed") -> torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_per_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """

    dual_weights = get_dual_area_weights_packed(Surfaces)

    if isinstance(Surfaces, trimesh.Trimesh):
        face_areas = torch.from_numpy(Surfaces.area_faces).unsqueeze(-1).float()
        face2vertex_index = Surfaces.faces.reshape(-1)
        face2vertex_index = torch.from_numpy(face2vertex_index).long()
    elif isinstance(Surfaces, Meshes):
        face_areas = Surfaces.faces_areas_packed().unsqueeze(-1)
        face2vertex_index = Surfaces.faces_packed().view(-1)
    else:
        raise ValueError("Surfaces must be a Meshes or a Trimesh object")

    dual_area_per_vertex = scatter(
        (dual_weights * face_areas).view(-1), face2vertex_index, reduce="sum"
    )

    return dual_area_per_vertex


def dual_gather_from_face_features_to_vertices_packed(
    mesh: Meshes, features_faces: torch.Tensor, mode="dual"
) -> torch.Tensor:
    """
    Gather face features to vertices with dual area weighting.
    Args:
        mesh: Meshes object representing a batch of meshes.
        features_faces: Tensor of shape (F, D) where F is the number of faces and D is the number of features.
        mode: str, either 'mean' or 'dual'.
    Returns:
        Tensor of shape (V, D) where V is the number of vertices.
    """
    if isinstance(mesh, Meshes):
        F = mesh.faces_packed().shape[0]
        face_2_vert_index = mesh.faces_packed()
        face_area = mesh.faces_areas_packed()
    elif isinstance(mesh, trimesh.Trimesh):
        F = mesh.faces.shape[0]
        face_2_vert_index = torch.tensor(mesh.faces, dtype=torch.long)
        face_area = torch.tensor(mesh.area_faces, dtype=torch.float32)

    D = features_faces.shape[-1]

    assert features_faces.shape[0] == F, (
        "features_faces must have the same number of faces as the mesh."
    )

    if isinstance(mesh, Meshes):
        F = mesh.faces_packed().shape[0]
        V = mesh.verts_packed().shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        F = mesh.faces.shape[0]
        V = mesh.vertices.shape[0]

    if mode == "mean":
        return scatter(
            repeat(features_faces, "F D -> (F V) D", V=3),
            face_2_vert_index,
            dim=0,
            reduce="mean",
            dim_size=V,
        )

    dual_area_face_splitted = get_dual_area_weights_packed(mesh) * (
        face_area.view(F, 1)
    )
    face_features_weighted = dual_area_face_splitted * features_faces

    face_2_vert_index = rearrange(face_2_vert_index, "F V -> (F V) 1", F=F, V=3)
    interal_features = scatter(
        face_features_weighted.view(-1, D),
        face_2_vert_index,
        dim=0,
        reduce="sum",
        dim_size=V,
    )
    dual_area_vert = get_dual_area_vertex_packed(mesh)

    reciprocal_dual_area_vert = dual_area_vert.reciprocal()

    reciprocal_dual_area_vert[torch.where(torch.isinf(reciprocal_dual_area_vert))] = 1e6

    return interal_features * (reciprocal_dual_area_vert.view(-1, 1))


def dual_interpolation_from_verts_to_faces_packed(
    mesh: Meshes, features_verts: torch.Tensor, mode="mean"
) -> torch.Tensor:
    """
    Interpolate vertex features to faces with dual area weighting.
    Args:
        mesh: Meshes object representing a batch of meshes.
        features_verts: Tensor of shape (V, D) where V is the number of vertices and D is the number of features.
        mode: str, either 'mean' or 'dual'.
    Returns:
        Tensor of shape (F, D) where F is the number of faces.
    """
    D = features_verts.shape[1]
    if isinstance(mesh, Meshes):
        faces = mesh.faces_packed()
        V = mesh.verts_packed().shape[0]
        F = mesh.faces_packed().shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        V = mesh.vertices.shape[0]
        F = faces.shape[0]
    assert features_verts.shape[0] == V, (
        "features_verts must have the same number of vertices as the mesh."
    )
    dual_weights = get_dual_area_weights_packed(mesh).view(F, 3, 1)

    features_verts_faces = features_verts[faces, :]
    if mode == "mean":
        return (features_verts_faces).mean(dim=1)
    elif mode == "dual":
        return (dual_weights * features_verts_faces).sum(dim=1)
