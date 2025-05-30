import torch
import trimesh
from pytorch3d.structures import Meshes


def get_soft_volume(mesh: Meshes):
    """
    Compute the (soft) volume of the mesh using the Gauss-Integral theorem.
    """
    if isinstance(mesh, Meshes):
        face_areas = mesh.faces_areas_packed()
        face_normals = mesh.faces_normals_packed()
        face_barycenters = mesh.verts_packed()[mesh.faces_packed()].mean(dim=1)
        volume_ele = (face_barycenters * face_normals).sum(dim=-1) * face_areas
        volume_ele_padded = face_feature_packed_padded(mesh, volume_ele.view(-1, 1))
        vol = volume_ele_padded.sum(dim=1) / 3
        vol = vol.view(-1)
    elif isinstance(mesh, trimesh.Trimesh):
        face_areas = torch.from_numpy(mesh.area_faces).float()
        face_normals = torch.from_numpy(mesh.face_normals).float()
        face_barycenters = torch.from_numpy(mesh.triangles_center).float()
        vol = ((face_barycenters * face_normals).sum(dim=-1) * face_areas).sum() / 3
    return vol
