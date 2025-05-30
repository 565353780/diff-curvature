import torch


def calc_face_normals(
    vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
    faces: torch.Tensor,  # F,3 long, first face may be all zero
    normalize: bool = False,
) -> torch.Tensor:  # F,3
    """
       n
       |
       c0     corners ordered counterclockwise when
      / \     looking onto surface (in neg normal direction)
    c1---c2
    """
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # F,3
    if normalize:
        face_normals = torch.nn.functional.normalize(
            face_normals, eps=1e-6, dim=1
        )  # TODO inplace?
    return face_normals  # F,3


def calc_vertex_normals(
    vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
    faces: torch.Tensor,  # F,3 long, first face may be all zero
    face_normals: torch.Tensor = None,  # F,3, not normalized
) -> torch.Tensor:  # F,3
    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)

    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device
    )  # V,C=3,3
    vertex_normals.scatter_add_(
        dim=0,
        index=faces[:, :, None].expand(F, 3, 3),
        src=face_normals[:, None, :].expand(F, 3, 3),
    )
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)


def compute_faces_view_normal(
    verts: torch.Tensor, faces: torch.Tensor, mv: torch.Tensor
):
    """
    Compute face normals in the view space using [mv].

    @ verts: [# point, 3]
    @ faces: [# face, 3]
    @ mv: [# batch, 4, 4]
    """
    faces_normals = calc_face_normals(verts, faces, True)  # [F, 3]
    faces_normals_hom = torch.cat(
        (faces_normals, torch.zeros_like(faces_normals[:, [1]])), dim=-1
    )  # [F, 4]
    faces_normals_hom = faces_normals_hom.unsqueeze(0).unsqueeze(-1)  # [1, F, 4, 1]
    e_mv = mv.unsqueeze(1)  # [B, 1, 4, 4]
    faces_normals_view = e_mv @ faces_normals_hom  # [B, F, 4, 1]
    faces_normals_view = faces_normals_view[:, :, :3, 0]  # [B, F, 3]
    faces_normals_view[faces_normals_view[..., 2] > 0] = -faces_normals_view[
        faces_normals_view[..., 2] > 0
    ]  # [B, F, 3]

    return faces_normals_view


def compute_faces_intense(
    verts: torch.Tensor, faces: torch.Tensor, mv: torch.Tensor, lightdir: torch.Tensor
):
    """
    Compute face intense using [mv] and [lightdir].

    @ verts: [# point, 3]
    @ faces: [# face, 3]
    @ mv: [# batch, 4, 4]
    @ lightdir: [# batch, 3]
    """
    faces_normals_view = compute_faces_view_normal(verts, faces, mv)  # [B, F, 3]
    faces_attr = torch.sum(
        lightdir.unsqueeze(1) * faces_normals_view, -1, keepdim=True
    )  # [B, F, 1]
    faces_attr = torch.clamp(faces_attr, min=0.0, max=1.0)  # [B, F, 1]
    faces_intense = faces_attr[..., 0]  # [B, F]

    return faces_intense
