import torch
import numpy as np
import nvdiffrast.torch as dr
from pytorch3d.structures import Meshes

from diff_curvature.Config.light import LIGHT_DIR
from diff_curvature.Method.mesh_geometry import (
    get_gaussian_curvature_vertices_packed,
    get_mean_curvature_vertices_packed,
)
from diff_curvature.Module.Renderer.normal import NormalRenderer


class AlphaRenderer(NormalRenderer):
    """
    Renderer that renders
    * normal
    * depth
    * shillouette
    """

    def __init__(
        self,
        mv: torch.Tensor,  # C,4,4
        proj: torch.Tensor,  # C,4,4
        image_size: "tuple[int,int]",
    ):
        super().__init__(mv, proj, image_size)
        self._mv = mv
        self._proj = proj
        self.eps = 1e-4

    def forward(
        self,
        verts: torch.Tensor,
        normals: torch.Tensor,
        faces: torch.Tensor,
        curv_rescale: float = 10.0,
    ) -> torch.Tensor:
        """
        Single pass without transparency.
        """
        V = verts.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat(
            (verts, torch.ones(V, 1, device=verts.device)), axis=-1
        )  # V,3 -> V,4
        verts_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4
        rast_out, _ = dr.rasterize(
            self._glctx, verts_clip, faces, resolution=self._image_size, grad_db=False
        )  # C,H,W,4

        # View-space normal
        vert_normals_hom = torch.cat(
            (normals, torch.zeros(V, 1, device=verts.device)), axis=-1
        )  # V,3 -> V,4
        vert_normals_view = vert_normals_hom @ self._mv.transpose(-2, -1)  # C,V,4
        vert_normals_view = vert_normals_view[..., :3]  # C,V,3
        vert_normals_view[vert_normals_view[..., 2] > 0.0] = -vert_normals_view[
            vert_normals_view[..., 2] > 0.0
        ]
        vert_normals_view = vert_normals_view.contiguous()

        # View-space light direction
        lightdir = torch.tensor(
            LIGHT_DIR, dtype=torch.float32, device=verts.device
        )  # 3
        lightdir = lightdir.view((1, 1, 1, 3))  # 1,1,1,3

        # Pixel normals in view space
        pixel_normals_hom, _ = dr.interpolate(normals, rast_out, faces)  # C,H,W,3
        pixel_normals_hom = pixel_normals_hom / torch.clamp(
            torch.norm(pixel_normals_hom, p=2, dim=-1, keepdim=True), min=1e-5
        )

        pixel_normals_view, _ = dr.interpolate(
            vert_normals_view, rast_out, faces
        )  # C,H,W,3
        pixel_normals_view = pixel_normals_view / torch.clamp(
            torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5
        )

        # Diffuse shading
        diffuse = torch.sum(lightdir * pixel_normals_view, -1, keepdim=True)  # C,H,W,1
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        diffuse = diffuse[..., [0, 0, 0]]  # C,H,W,3

        # Depth
        verts_clip_w = verts_clip[..., [3]]
        verts_clip_w[
            torch.logical_and(verts_clip_w >= 0.0, verts_clip_w < self.eps)
        ] = self.eps
        verts_clip_w[
            torch.logical_and(verts_clip_w < 0.0, verts_clip_w > -self.eps)
        ] = -self.eps

        verts_depth = verts_clip[..., [2]] / verts_clip_w  # C,V,1
        depth, _ = dr.interpolate(verts_depth, rast_out, faces)  # C,H,W,1
        depth = (depth + 1.0) * 0.5  # Normalize depth to [0, 1]

        depth[rast_out[..., -1] == 0] = 1.0  # Exclude background
        depth = 1 - depth  # Invert depth for visualization
        max_depth = depth.max()
        min_depth = depth[depth > 0.0].min()  # Exclude background
        depth_info = {"raw": depth, "max": max_depth, "min": min_depth}

        # Silhouette (alpha)
        alpha = torch.clamp(rast_out[..., [-1]], max=1)  # C,H,W,1

        # Convert normals to RGB
        normals_rgb_hom = (
            pixel_normals_hom + 1.0
        ) * 0.5  # Shift and scale normals from [-1, 1] to [0, 1]

        # Curvature
        with torch.no_grad():
            meshes = Meshes(verts=[verts], faces=[faces])
            gs_curvatures_vert = get_gaussian_curvature_vertices_packed(meshes).view(
                -1, 1
            )  # V,1
            mean_curvature_vert = get_mean_curvature_vertices_packed(meshes).view(
                -1, 1
            )  # V,1
            total_curvature_vert = mean_curvature_vert**2 - 2 * gs_curvatures_vert

            gs_curvatures_vert = torch.tanh(
                gs_curvatures_vert / curv_rescale**2
            )  # C,H,W,1
            gs_curvatures_vert = (gs_curvatures_vert + 1.0) * 0.5
            gs_curvatures, _ = dr.interpolate(
                gs_curvatures_vert, rast_out, faces
            )  # C,H,W,1
            gs_curvatures = torch.clamp(gs_curvatures, min=0.0, max=1.0)

            mean_curvature_vert = torch.tanh(
                mean_curvature_vert / curv_rescale
            )  # C,H,W,1
            mean_curvature_vert = torch.exp(mean_curvature_vert) / np.exp(1.0)
            mean_curvature, _ = dr.interpolate(
                mean_curvature_vert, rast_out, faces
            )  # C,H,W,1
            mean_curvature = torch.clamp(mean_curvature, min=0.0, max=1.0)

            total_curvature_vert = torch.tanh(
                total_curvature_vert / curv_rescale**2
            )  # C,H,W,1
            total_curvature, _ = dr.interpolate(
                total_curvature_vert, rast_out, faces
            )  # C,H,W,1
            total_curvature = torch.clamp(total_curvature, min=0.0, max=1.0)

        # Combine output: diffuse, depth, alpha, and normals (in RGB)
        col = torch.concat(
            (
                diffuse,
                depth,
                alpha,
                normals_rgb_hom,
                gs_curvatures,
                mean_curvature,
                total_curvature,
            ),
            dim=-1,
        )
        col = dr.antialias(col, rast_out, verts_clip, faces)  # C,H,W,8

        return col, depth_info
