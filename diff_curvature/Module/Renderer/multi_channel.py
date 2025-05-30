import torch

from diff_curvature.Method.mesh_normal import calc_vertex_normals
from diff_curvature.Module.Renderer.alpha import AlphaRenderer


class MultiChannelRenderer:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, device: str):
        # geometry info;
        self.gt_vertices = verts
        self.gt_faces = faces
        self.gt_vertex_normals = calc_vertex_normals(verts, faces)

        # rendered images;
        self.gt_images = None
        self.gt_depth_info = None

    def render(self, renderer: AlphaRenderer):
        target_images, target_depth_info = renderer.forward(
            self.gt_vertices, self.gt_vertex_normals, self.gt_faces
        )

        self.gt_images = target_images
        self.gt_depth_info = target_depth_info

        return self.gt_images

    def diffuse_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., :3]

    def depth_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., [3, 3, 3]]

    def shillouette_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., [4, 4, 4]]

    def normal_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., 5:8]

    def gs_curvatures_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., [8, 8, 8]]

    def mean_curvature_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., [9, 9, 9]]

    def total_curvature_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., [10, 10, 10]]

    def curvatures_rgb_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[..., [10, 9, 8]]
