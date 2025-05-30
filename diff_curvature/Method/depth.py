import torch


def compute_verts_depth(verts: torch.Tensor, mv: torch.Tensor, proj: torch.Tensor):
    """
    Compute depth for each vertices using [mv, proj].

    @ verts: [# point, 3]
    @ mv: [# batch, 4, 4]
    @ proj: [# batch, 4, 4]
    """

    verts_hom = torch.cat((verts, torch.ones_like(verts[:, [0]])), dim=-1)  # [V, 4]
    verts_hom = verts_hom.unsqueeze(0).unsqueeze(-1)  # [1, V, 4, 1]
    e_mv = mv.unsqueeze(1)  # [B, 1, 4, 4]
    e_proj = proj.unsqueeze(1)  # [B, 1, 4, 4]
    verts_view = e_mv @ verts_hom  # [B, V, 4, 1]
    verts_proj = e_proj @ verts_view  # [B, V, 4, 1]
    verts_proj_w = verts_proj[..., [3], 0]  # [B, V, 1]

    # clamp w;
    verts_proj_w[torch.logical_and(verts_proj_w >= 0.0, verts_proj_w < 1e-4)] = 1e-4
    verts_proj_w[torch.logical_and(verts_proj_w < 0.0, verts_proj_w > -1e-4)] = -1e-4

    verts_ndc = verts_proj[..., :3, 0] / verts_proj_w  # [B, V, 3]
    verts_depth = verts_ndc[..., 2]  # [B, V]

    return verts_depth
