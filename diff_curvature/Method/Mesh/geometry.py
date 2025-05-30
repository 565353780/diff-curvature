import torch
import trimesh
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.ops import packed_to_padded


def normalize_mesh(mesh, rescalar=0.99):
    """
    Normalize the mesh to fit in the unit sphere
    Args:
        mesh: Meshes object or trimesh object
        rescalar: float, the scale factor to rescale the mesh
    """
    if isinstance(mesh, Meshes):
        bbox = mesh.get_bounding_boxes()
        B = bbox.shape[0]
        center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
        center = center.view(B, 1, 3)
        size = bbox[:, :, 1] - bbox[:, :, 0]

        scale = 2.0 / (torch.max(size, dim=1)[0] + 1e-8).view(B, 1) * rescalar
        scale = scale.view(B, 1, 1)
        mesh = mesh.update_padded((mesh.verts_padded() - center) * scale)
        return mesh

    elif isinstance(mesh, trimesh.Trimesh):
        bbox_min, bbox_max = mesh.bounds
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min

        # Scale factor to normalize to [-1, 1]
        scale_factor = 2.0 / np.max(
            bbox_size
        )  # Ensures the longest side fits in [-1, 1]

        # Apply translation and scaling
        mesh.apply_translation(-bbox_center)  # Move the mesh center to the origin
        mesh.apply_scale(scale_factor)

    return mesh


def get_faces_coordinates_padded(meshes: Meshes):
    """
    Get the faces coordinates of the meshes in padded format.
    return:
        face_coord_padded: [B, F, 3, 3]
    """
    face_mesh_first_idx = meshes.mesh_to_faces_packed_first_idx()
    face_coord_packed = get_faces_coordinates_packed(
        meshes.verts_packed(), meshes.faces_packed()
    )

    face_coord_padded = packed_to_padded(
        face_coord_packed, face_mesh_first_idx, max_size=meshes.faces_padded().shape[1]
    )

    return face_coord_padded


def get_faces_coordinates_packed(*args):
    """
    Get the faces coordinates of the meshes in padded format.
    return:
        face_coord_padded: [F, 3, 3]
    """
    if len(args) == 1:
        if isinstance(args[0], Meshes):
            vertices_packed = args[0].verts_packed()
            faces_packed = args[0].faces_packed()
        elif isinstance(args[0], trimesh.Trimesh):
            vertices_packed = args[0].vertices.astype(np.float32)
            faces_packed = args[0].faces.astype(np.int64)

    elif len(args) == 2:
        vertices_packed = args[0]
        faces_packed = args[1]

    face_coord_packed = vertices_packed[faces_packed, :]

    return face_coord_packed


def get_faces_angle_packed(*args):
    """
    Compute the angle of each face.
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """

    if len(args) == 1:
        if isinstance(args[0], trimesh.Trimesh):
            return args[0].face_angles

    Face_coord = get_faces_coordinates_packed(*args)

    if not isinstance(Face_coord, torch.Tensor):
        Face_coord = torch.tensor(Face_coord, dtype=torch.float32)

    A = Face_coord[:, 1, :] - Face_coord[:, 0, :]
    B = Face_coord[:, 2, :] - Face_coord[:, 1, :]
    C = Face_coord[:, 0, :] - Face_coord[:, 2, :]

    angle_0 = torch.arccos(
        -torch.sum(A * C, dim=1)
        / (1e-10 + (torch.norm(A, dim=1) * torch.norm(C, dim=1)))
    )
    angle_1 = torch.arccos(
        -torch.sum(A * B, dim=1)
        / (1e-10 + (torch.norm(A, dim=1) * torch.norm(B, dim=1)))
    )
    angle_2 = torch.arccos(
        -torch.sum(B * C, dim=1)
        / (1e-10 + (torch.norm(B, dim=1) * torch.norm(C, dim=1)))
    )
    angles = torch.stack([angle_0, angle_1, angle_2], dim=1)

    if len(args) == 2 and not isinstance(args[1], torch.Tensor):
        angles = angles.detach().cpu().numpy()

    return angles
